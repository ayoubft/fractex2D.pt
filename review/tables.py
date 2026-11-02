"""Aggregate review.evaluate JSONs into the report tables (CSV + markdown) and PR plot.

Usage:
    python -m review.tables --metrics review_out/metrics --out review_experiments \
        --row "U-Net=unet_sel" --row "SAM 2 + FPN=sam2_tiny_s0,sam2_tiny_s1,sam2_tiny_s2" ...
        [--compute review_out/compute.json --compute-row "U-Net=unet_sel" ...]
        [--train-time "SAM 2 + FPN=outputs_review/sam2_tiny_s0,..."]

Rows with several tags are reported as mean ± sample std (ddof=1) over the tags;
single-tag rows are marked "1 seed". Metrics in [0,1] get 3 decimals, PSNR 1,
FracSim 2 (it is an unbounded chi-square distance, lower is better).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLS = ["mse", "psnr", "ssim", "prec", "rec", "f1", "iou", "ap", "fracsim"]
HEAD = {"mse": "MSE", "psnr": "PSNR", "ssim": "SSIM", "prec": "P", "rec": "R", "f1": "F1",
        "iou": "IoU", "ap": "AP", "fracsim": "FracSim"}
THRESHOLDS = ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]


def fmt(col, vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "n/a"
    d = 1 if col == "psnr" else 2 if col == "fracsim" else 3
    if len(vals) == 1:
        return f"{vals[0]:.{d}f}"
    return f"{np.mean(vals):.{d}f} ± {np.std(vals, ddof=1):.{d}f}"


def row_metrics(m, variant, thr_key, group="all", agg="pooled"):
    """One model's metrics at a threshold key ('0.1' or 'sel')."""
    t = f"{m['selected_threshold'][variant]:g}" if thr_key == "sel" else thr_key
    reg = m["test"]["regression"][group]
    src = m["test"]["pooled"] if agg == "pooled" else m["test"]["per_image"]
    cls = src[variant][group][t]
    fs = m["test"].get("fracsim", {}).get(variant, {}).get(group, {}).get(t)
    return {"mse": reg["mse"], "psnr": reg["psnr"], "ssim": reg["ssim"], "prec": cls["prec"],
            "rec": cls["rec"], "f1": cls["f1"], "iou": cls["iou"], "ap": m["test"]["ap"][variant][group],
            "fracsim": fs, "threshold": float(t)}


def write_csv(path, header, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("review_out/metrics"))
    ap.add_argument("--out", type=Path, default=Path("review_experiments"))
    ap.add_argument("--row", action="append", required=True, help="LABEL=tag1,tag2,...")
    ap.add_argument("--compute", type=Path)
    ap.add_argument("--compute-row", action="append", default=[], help="LABEL=compute_tag")
    ap.add_argument("--train-time", action="append", default=[], help="LABEL=run_dir1,run_dir2 (epoch_log.csv)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.row:
        label, tags = spec.split("=", 1)
        ms = [json.loads((args.metrics / f"{t}.json").read_text()) for t in tags.split(",")]
        rows.append((label, tags.split(","), ms))
    sites = rows[0][2][0]["sites"]
    md = []

    def seeds_note(ms):
        return f"{len(ms)} seeds" if len(ms) > 1 else "1 seed"

    # --- main tables
    for variant in ("dilated", "original"):
        for agg in ("pooled", "per_image"):
            for thr in ("0.1", "sel"):
                header = ["Model", "n"] + [HEAD[c] for c in COLS] + (["threshold (val-selected)"] if thr == "sel" else [])
                body = []
                for label, tags, ms in rows:
                    vals = [row_metrics(m, variant, thr, agg=agg) for m in ms]
                    r = [label, seeds_note(ms)] + [fmt(c, [v[c] for v in vals]) for c in COLS]
                    if thr == "sel":
                        r.append(", ".join(f"{v['threshold']:g}" for v in vals))
                    body.append(r)
                name = f"main_{variant}_{agg}_{'t0.1' if thr == '0.1' else 'valsel'}"
                write_csv(args.out / f"{name}.csv", header, body)
                md.append(f"### {name}\n\n" + md_table(header, body))

    # --- threshold sensitivity (pooled, test), both target variants
    for variant in ("dilated", "original"):
        header = ["Model"] + [f"{k}@{t}" for t in THRESHOLDS for k in ("F1", "IoU")]
        body = []
        for label, tags, ms in rows:
            r = [label]
            for t in THRESHOLDS:
                for k in ("f1", "iou"):
                    r.append(fmt(k, [m["test"]["pooled"][variant]["all"][t][k] for m in ms]))
            body.append(r)
        write_csv(args.out / f"threshold_sensitivity_{variant}.csv", header, body)
        md.append(f"### threshold_sensitivity_{variant} (pooled, test)\n\n" + md_table(header, body))

    # --- per site (pooled, test, dilated + original, t=0.1 and val-selected)
    for variant in ("dilated", "original"):
        for thr in ("0.1", "sel"):
            header = ["Model", "Site"] + [HEAD[c] for c in COLS]
            body = []
            for label, tags, ms in rows:
                for s in sites:
                    vals = [row_metrics(m, variant, thr, group=s) for m in ms]
                    body.append([label, s] + [fmt(c, [v[c] for v in vals]) for c in COLS])
            name = f"per_site_{variant}_{'t0.1' if thr == '0.1' else 'valsel'}"
            write_csv(args.out / f"{name}.csv", header, body)
            md.append(f"### {name} (pooled, test)\n\n" + md_table(header, body))

    # --- per-seed raw numbers
    header = ["Model", "tag", "checkpoint_md5", "variant", "agg", "threshold"] + [HEAD[c] for c in COLS] + ["val_loss"]
    body = []
    for label, tags, ms in rows:
        for tag, m in zip(tags, ms):
            for variant in ("dilated", "original"):
                for agg in ("pooled", "per_image"):
                    for thr in ("0.1", "sel"):
                        v = row_metrics(m, variant, thr, agg=agg)
                        body.append([label, tag, m["meta"]["checkpoint_md5"], variant, agg, f"{v['threshold']:g}"]
                                    + [("" if v[c] is None else f"{v[c]:.6g}") for c in COLS]
                                    + [f"{m['meta']['splits']['val']['loss']:.6g}"])
    write_csv(args.out / "per_seed_raw.csv", header, body)

    # --- PR curves (test, pooled, all sites)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    pr_rows = []
    for ax, variant in zip(axes, ("dilated", "original")):
        for label, tags, ms in rows:
            for k, tag in enumerate(tags):
                pr = np.load(args.metrics / f"{tag}.pr_test.npz")[variant]
                ax.plot(pr[:, 2], pr[:, 1], lw=1.2, alpha=0.9 if k == 0 else 0.45,
                        label=f"{label} (AP {fmt('ap', [m['test']['ap'][variant]['all'] for m in ms])})" if k == 0 else None,
                        color=f"C{[r[0] for r in rows].index(label)}")
                pr_rows += [[label, tag, variant, f"{t:.6g}", f"{p:.6g}", f"{r:.6g}"] for t, p, r in pr]
        ax.set_title(f"test, pooled, target = {variant} (> 0)")
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(args.out / "pr_curves.png", dpi=160)
    write_csv(args.out / "pr_curves.csv", ["Model", "tag", "target", "threshold", "precision", "recall"], pr_rows)

    # --- compute
    if args.compute and args.compute.is_file():
        comp = json.loads(args.compute.read_text())
        times = {}
        for spec in args.train_time:
            label, dirs = spec.split("=", 1)
            per_epoch, totals = [], []
            for d in dirs.split(","):
                f = Path(d) / "epoch_log.csv"
                if f.is_file():
                    secs = np.loadtxt(f, delimiter=",", skiprows=1, usecols=3, ndmin=1)
                    per_epoch.append(secs.mean())
                    totals.append(secs.sum())
            times[label] = (per_epoch, totals)
        header = ["Model", "Params (M)", "GFLOPs (256x256)", "Latency b1 (ms/img)", "Latency b8 (ms/img)",
                  "Peak mem infer b8 (MiB)", "Peak mem train b8 (MiB)", "Time/epoch (min)", "Total train (h)"]
        body = []
        for spec in args.compute_row:
            label, tag = spec.split("=", 1)
            c = comp["models"][tag]
            pe, tot = times.get(label, ([], []))
            body.append([label, f"{c['params_M']:.1f}", f"{c['gflops']:.1f}",
                         f"{c['latency_b1_ms_per_image']:.2f}", f"{c['latency_b8_ms_per_image']:.2f}",
                         f"{c['peak_mem_infer_b8_MiB']:.0f}", f"{c['peak_mem_train_b8_MiB']:.0f}",
                         fmt("psnr", [x / 60 for x in pe]) if pe else "see report",
                         fmt("psnr", [x / 3600 for x in tot]) if tot else "see report"])
        write_csv(args.out / "compute.csv", header, body)
        md.append(f"### compute (GPU: {comp['gpu']})\n\n" + md_table(header, body))

    (args.out / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
