"""Run a trained checkpoint over val/test deterministically and save probability maps.

Usage:
    python -m review.predict <run_dir> --tag NAME [--checkpoint model.pt]
        [--splits val test] [--out-root review_out/preds] [--no-save]
        [hydra.overrides ...]

The model and data pipeline are rebuilt from <run_dir>/.hydra/config.yaml (as in
infere.py). `dataset.deterministic_eval=true` is always forced, so val/test see no
random flips and every call on the same checkpoint gives the same numbers. Extra
overrides such as `++dataset.dem_mode=raw` are passed through.

Writes to <out-root>/<tag>/:
    <split>_pred.npy   float16 (N, H, W), the model's raw output in [0, 1]
    meta.json          checkpoint md5, overrides, per-split loss (the run's own
                       training criterion), timing
With --no-save only meta.json is written (used for validation-loss selection).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infere import load_cfg  # noqa: E402


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--out-root", type=Path, default=Path("review_out/preds"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args, overrides = ap.parse_known_args(sys.argv[1:] if argv is None else argv)

    overrides = ["++dataset.deterministic_eval=true", *overrides]
    cfg = load_cfg(args.run_dir, overrides)
    model_name = cfg.model._target_.split(".")[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.run_dir / args.checkpoint
    model = instantiate(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    criterion = instantiate(cfg.loss)

    _, valloader, testloader = instantiate(cfg.dataset)
    loaders = {"val": valloader, "test": testloader}

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "tag": args.tag, "run_dir": str(args.run_dir.resolve()), "checkpoint": args.checkpoint,
        "checkpoint_md5": md5(ckpt), "model": model_name, "in_channels": cfg.in_channels,
        "overrides": overrides, "datasets": cfg.dataset.datasets,
        "dem_mode": cfg.dataset.get("dem_mode", "legacy"),
        "loss_fn": cfg.loss._target_, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "splits": {},
    }

    for split in args.splits:
        ds = loaders[split].dataset
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        n = len(ds)
        preds = None
        if not args.no_save:
            preds = np.lib.format.open_memmap(out_dir / f"{split}_pred.npy", mode="w+",
                                              dtype=np.float16, shape=(n, cfg.dataset.shape, cfg.dataset.shape))
        loss_sum, i0, t0 = 0.0, 0, time.time()
        lo, hi = float("inf"), float("-inf")
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"{args.tag}:{split}"):
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                out = model(imgs)
                out = out["out"] if isinstance(out, dict) else out
                loss_sum += criterion(out, labels).item() * imgs.shape[0]
                lo, hi = min(lo, out.min().item()), max(hi, out.max().item())
                if preds is not None:
                    preds[i0:i0 + imgs.shape[0]] = out[:, 0].float().cpu().numpy().astype(np.float16)
                i0 += imgs.shape[0]
        if preds is not None:
            preds.flush()
            del preds
        # outputs must already be probabilities; logits would break every metric
        meta["splits"][split] = {"n": n, "loss": loss_sum / n, "seconds": round(time.time() - t0, 1),
                                 "out_min": lo, "out_max": hi}
        print(f"{args.tag} {split}: n={n} loss={loss_sum / n:.6f}")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote", out_dir / "meta.json")


if __name__ == "__main__":
    main()
