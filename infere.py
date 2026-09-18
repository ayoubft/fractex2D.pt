"""Run a trained checkpoint over the benchmark test set and over full rasters.

Usage:
    python infere.py <run_dir> [options] [config.overrides=...]

<run_dir> is a Hydra output directory produced by main.py, e.g.
`outputs_BM_/sam2_tiny_dicebce/2026-09-17_10-30`. The model is rebuilt from
that run's own saved config (`<run_dir>/.hydra/config.yaml`), so inference
always matches how the checkpoint was trained.

Examples:
    python infere.py outputs_BM_/sam2_tiny_dicebce/2026-09-17_10-30
    python infere.py outputs_BM_/sam2_large/2026-09-18_02-11 --checkpoint final_model.pt
    python infere.py outputs_BM_/unet/2026-01-29_09-54 --no-raster threshold=0.3
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io
from torchmetrics.classification import (
    BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall,
)
from tqdm.auto import tqdm

from src.train2 import eval_loop

# Full rasters to predict on. Each entry <p> expects <p>.png, plus <p>-dem.tif
# when the run was trained with in_channels=4.
DEFAULT_IMAGES = [
    "data/test_ovas/kl5/kl5-s3",
    "data/test_ovas/kl5/hnn-z1-s3",
    "data/test_ovas/kl5/_matteo21-z1-s3",
    "data/test_ovas/kl5/ortho-ldb-z1-s3",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Hydra output dir from main.py (must contain .hydra/config.yaml)")
    parser.add_argument(
        "--checkpoint", default="model.pt",
        help="Checkpoint file inside run_dir. 'model.pt' is the best-validation-loss "
             "checkpoint that main.py reports its test metrics from; 'final_model.pt' "
             "is the last epoch. (default: %(default)s)")
    parser.add_argument(
        "--images", nargs="+", default=DEFAULT_IMAGES,
        help="Raster stems to predict (default: the four ovaskainen test rasters)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write predictions and metrics (default: run_dir)")
    parser.add_argument(
        "--no-raster", action="store_true",
        help="Only evaluate the test set; skip full-raster prediction")
    parser.add_argument(
        "--sweep-thresholds", action="store_true",
        help="Score a grid of thresholds in a single forward pass and report the "
             "best by F1 and by IoU. Implies --no-raster.")
    parser.add_argument(
        "--threshold-grid", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
        default=[0.05, 0.95, 0.05],
        help="Threshold grid for --sweep-thresholds, inclusive of STOP "
             "(default: %(default)s)")
    parser.add_argument(
        "--split", choices=["test", "val"], default="test",
        help="Which split to sweep thresholds on. 'test' matches the ODS convention "
             "and the existing FscoreODS.py; 'val' avoids tuning on the split you "
             "report. (default: %(default)s)")
    # anything unrecognised is passed through to Hydra as a config override
    return parser.parse_known_args(argv)


def load_cfg(run_dir: Path, overrides):
    """Rebuild the config the run was trained with, from its own .hydra dir."""
    hydra_dir = (run_dir / ".hydra").resolve()
    if not (hydra_dir / "config.yaml").is_file():
        raise SystemExit(f"No saved config at {hydra_dir / 'config.yaml'} -- "
                         f"is {run_dir} really a main.py output dir?")
    with initialize_config_dir(config_dir=str(hydra_dir), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def sweep_thresholds(model, loader, thresholds, device, model_name):
    """Score every threshold from a single forward pass over `loader`.

    The model's output does not depend on the threshold -- only the binarisation
    of it does -- so running the network once and reusing the probabilities is
    exactly equivalent to re-running inference per threshold, at 1/N the cost.

    Binarisation matches src/train2.py eval_loop: strictly `> threshold` for the
    prediction and `> 0.` for the label, so these numbers are directly
    comparable with the ones main.py reports.
    """
    model.eval()
    metrics = {
        t: {"f1": BinaryF1Score().to(device),
            "iou": BinaryJaccardIndex().to(device),
            "prec": BinaryPrecision().to(device),
            "rec": BinaryRecall().to(device)}
        for t in thresholds
    }

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Threshold sweep"):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)['out'] if model_name == 'fcn_resnet101' else model(imgs)
            if model_name == 'Segformer':          # same quirk as eval_loop
                out[out > 0.99] = 0.
            labels_clf = (labels > 0.).float()
            for t, ms in metrics.items():
                pred_clf = (out > t).float()
                for m in ms.values():
                    m(pred_clf, labels_clf)

    rows = []
    for t in thresholds:
        rows.append({"threshold": round(float(t), 4),
                     **{k: float(m.compute()) for k, m in metrics[t].items()}})
    return rows


def predict_raster(model, img_stem, cfg, device):
    """Patch-based prediction over one full raster. Returns a uint8 probability map."""
    img = io.imread(f"{img_stem}.png")

    if cfg.in_channels == 4:
        dem = io.imread(f"{img_stem}-dem.tif").astype(np.float32)
        # Min-max, matching BaseCrackDataset and main.py. The previous version
        # z-scored the DEM here, which did not match how the model was trained.
        dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
        combined = np.concatenate((img[:, :, :3], np.expand_dims(dem, 2)), axis=2)
    else:
        # 3-channel models (e.g. SAM2) predict from RGB alone.
        combined = img[:, :, :3]

    patch_size = cfg.dataset.shape
    h, w, c = combined.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded = np.pad(combined, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    patches = patchify(padded, (patch_size, patch_size, c), step=patch_size)

    pred_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = torch.Tensor(patches[i, j]).permute(0, 3, 1, 2) / 255.0
            with torch.no_grad():
                pred_patches.append(model(patch.to(device)).cpu())

    pred = np.reshape(pred_patches,
                      (patches.shape[0], patches.shape[1], 1, patch_size, patch_size, 1))
    pred = unpatchify(pred, padded.shape[:2] + (1,))
    return (pred[:h, :w, :].reshape(h, w) * 255).astype(np.uint8)


def main(argv=None):
    args, overrides = parse_args(sys.argv[1:] if argv is None else argv)

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(run_dir, overrides)
    model_name = cfg.model._target_.split(".")[-1]

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
    print(f"Run dir     : {run_dir}")
    print(f"Model       : {model_name} (in_channels={cfg.in_channels})")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Device      : {device}")

    ckpt = run_dir / args.checkpoint
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model = instantiate(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    criterion = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)

    _, valloader, testloader = instantiate(cfg.dataset)

    # --- Threshold sweep ---
    if args.sweep_thresholds:
        start, stop, step = args.threshold_grid
        grid = np.round(np.arange(start, stop + step / 2, step), 4)
        loader = testloader if args.split == "test" else valloader
        print(f"Sweeping {len(grid)} thresholds on the {args.split} split "
              f"({grid[0]:.2f} to {grid[-1]:.2f} step {step:g}), single forward pass")

        rows = sweep_thresholds(model, loader, grid, device, model_name)

        print(f"\n{'threshold':>10} {'f1':>8} {'iou':>8} {'prec':>8} {'rec':>8}")
        for r in rows:
            print(f"{r['threshold']:>10.2f} {r['f1']:>8.4f} {r['iou']:>8.4f} "
                  f"{r['prec']:>8.4f} {r['rec']:>8.4f}")

        best_f1 = max(rows, key=lambda r: r["f1"])
        best_iou = max(rows, key=lambda r: r["iou"])
        print(f"\nbest F1  : {best_f1['f1']:.4f} at threshold {best_f1['threshold']:.2f}")
        print(f"best IoU : {best_iou['iou']:.4f} at threshold {best_iou['threshold']:.2f}")
        print(f"(config threshold is {cfg.threshold})")

        stem = Path(args.checkpoint).stem
        csv_file = out_dir / f"threshold_sweep_{args.split}_{stem}.csv"
        with csv_file.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote", csv_file)

        json_file = out_dir / f"threshold_sweep_{args.split}_{stem}.json"
        json_file.write_text(json.dumps(
            {"run_dir": str(run_dir), "checkpoint": args.checkpoint,
             "model": model_name, "split": args.split,
             "datasets": cfg.dataset.datasets, "config_threshold": cfg.threshold,
             "best_f1": best_f1, "best_iou": best_iou, "grid": rows}, indent=2))
        print("Wrote", json_file)
        return

    # --- Evaluate on the test set ---
    test_metrics = eval_loop(model, scheduler, criterion, testloader,
                             cfg.threshold, device, model_name, cfg.ignore_index)
    print(f"Datasets    : {cfg.dataset.datasets}")
    print("Test metrics:", test_metrics)

    metrics_file = out_dir / f"test_metrics_{Path(args.checkpoint).stem}.json"
    metrics_file.write_text(json.dumps(
        {"run_dir": str(run_dir), "checkpoint": args.checkpoint, "model": model_name,
         "in_channels": cfg.in_channels, "threshold": cfg.threshold,
         "datasets": cfg.dataset.datasets, "metrics": test_metrics},
        indent=2))
    print("Wrote", metrics_file)

    # --- Predict full rasters ---
    if args.no_raster:
        return

    for img_stem in args.images:
        if not Path(f"{img_stem}.png").is_file():
            print(f"Skipping missing raster: {img_stem}.png")
            continue
        pred = predict_raster(model, img_stem, cfg, device)
        dest = out_dir / f"{cfg.dataset.datasets}_pred_{Path(img_stem).name}.png"
        Image.fromarray(pred).save(dest)
        print("Wrote", dest)


if __name__ == "__main__":
    main()
