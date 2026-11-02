"""Dump the val/test targets once, in the exact order review.predict sees them.

Usage:
    python -m review.dump_targets [--out review_out/targets] [--datasets samsu19-matteo21-ovaskainen23]

Three target definitions, all built by the training pipeline itself
(src.dataset_benchm.all_datasets with deterministic_eval=True):

    dilated   expand=True,  dilate=True   the soft training target (uint8 = target*255).
                                          The paper's metrics binarise it with > 0.
    expanded  expand=True,  dilate=False  label refinement only (wide dark fractures
                                          merged into the annotation), no dilation halo.
    original  expand=False, dilate=False  the annotation as stored on disk.

Also writes <split>_site.npy (index into SITES) and <split>_names.txt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_benchm import all_datasets  # noqa: E402

VARIANTS = {"dilated": (True, True), "expanded": (True, False), "original": (False, False)}
SITE_OF = {"samsu19": "Australia", "matteo21": "Arizona", "ovaskainen23": "Finland"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("review_out/targets"))
    ap.add_argument("--datasets", default="samsu19-matteo21-ovaskainen23")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    keys = args.datasets.split("-")
    sites = [SITE_OF[k] for k in keys]

    info = {"datasets": args.datasets, "sites": sites, "variants": {}}
    for variant, (expand, dilate) in VARIANTS.items():
        _, valloader, testloader = all_datasets(batch_size=32, datasets=args.datasets, in_channels=3,
                                                expand=expand, dilate=dilate, deterministic_eval=True)
        for split, loader in (("val", valloader), ("test", testloader)):
            concat = loader.dataset
            n = len(concat)
            arr = np.lib.format.open_memmap(args.out / f"{split}_{variant}.npy", mode="w+",
                                            dtype=np.uint8, shape=(n, 256, 256))
            i0 = 0
            for _, masks in tqdm(DataLoader(concat, batch_size=64, num_workers=args.workers),
                                 desc=f"{variant}:{split}"):
                m = masks[:, 0].numpy()
                arr[i0:i0 + len(m)] = np.rint(m * 255).astype(np.uint8)
                i0 += len(m)
            arr.flush()
            vals = np.unique(arr[: min(n, 400)])
            info["variants"].setdefault(variant, {})[split] = {"n": n, "values_sample": vals.tolist(),
                                                               "positive_frac": float((arr > 0).mean())}
            if variant == "dilated":
                site = np.concatenate([np.full(len(d), i, dtype=np.int8) for i, d in enumerate(concat.datasets)])
                np.save(args.out / f"{split}_site.npy", site)
                names = [f"{keys[i]}/{Path(p).name}" for i, d in enumerate(concat.datasets) for p in d.masks]
                (args.out / f"{split}_names.txt").write_text("\n".join(names) + "\n")
                info.setdefault("counts", {})[split] = {sites[i]: len(d) for i, d in enumerate(concat.datasets)}
            del arr
    (args.out / "targets_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
