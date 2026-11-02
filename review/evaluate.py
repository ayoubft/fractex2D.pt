"""Score one model's saved val/test probability maps under the review protocol.

Usage:
    python -m review.evaluate review_out/preds/<tag> [--targets review_out/targets]
        [--out review_out/metrics/<tag>.json]

Protocol (reviewer response, Paper #47):
  * Inputs are the float16 maps written by review.predict (deterministic eval).
  * Regression metrics (MSE, PSNR, SSIM) compare the probability map with the soft
    `dilated` target in [0, 1]. MSE is pooled over all pixels and PSNR = 10 log10(1/MSE)
    of that pooled MSE (what torchmetrics' accumulated PSNR does). SSIM is the mean of
    per-image SSIM (11x11 Gaussian, sigma 1.5) with data_range fixed to 1.0; the training
    code let torchmetrics infer the range per batch.
  * Classification metrics binarise the prediction with `> t` and each target variant
    with `> 0` (see review.dump_targets): `dilated` reproduces the paper, `original` is the
    stored annotation, `expanded` is refinement without the dilation halo.
      - pooled: one confusion matrix over all pixels of the split (or site);
      - per-image mean: metrics per image, averaged over the images where they are
        defined (0/0 images are skipped; the counts used are reported).
  * AP: area under the pooled precision-recall curve, step-wise (sklearn's definition),
    computed exactly from histograms of the float16 scores.
  * Threshold selection: the t on a 0.005 grid that maximises pooled F1 on VALIDATION,
    chosen separately for each target variant. Test is never used to choose anything.
  * FracSim: the FraXet benchmark definition (src/train2.py::fracture_similarity, same
    code as the public FraXet repo): skeletonise prediction and target, remove junction
    neighbourhoods, label segments (8-connectivity), histogram the segment lengths into
    np.linspace(0, 260, 20) bins, chi-square distance, mean over images. A distance:
    lower is better.

Note: thresholds are applied to the float16-rounded scores, so a pixel whose float32
score lies within float16 rounding of t can fall on the other side compared with the
training code. This shifts counts by a negligible amount.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.measure import label
from skimage.morphology import skeletonize
from torchmetrics.functional.image import structural_similarity_index_measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VARIANTS = ("dilated", "expanded", "original")
FIXED_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
GRID = np.round(np.arange(0.005, 1.0, 0.005), 3)
NBINS = 0x3C00 + 1  # float16 bit patterns 0 .. 1.0
BIN_VALUES = np.arange(NBINS, dtype=np.uint16).view(np.float16).astype(np.float64)
CHUNK = 64


# ---------------------------------------------------------------- FracSim (FraXet)
def remove_junctions(skel: np.ndarray) -> np.ndarray:
    """Vectorised, exact equivalent of src.train2.remove_junctions."""
    skel = skel.astype(np.uint8)
    h, w = skel.shape
    s = sum(skel[di:di + h - 2, dj:dj + w - 2].astype(np.int16) for di in range(3) for dj in range(3))
    junction = s > 4
    mask = np.zeros_like(skel, dtype=bool)
    for di in range(3):
        for dj in range(3):
            mask[di:di + h - 2, dj:dj + w - 2] |= junction
    return skel * (~mask)


def segment_hist(binary: np.ndarray) -> np.ndarray:
    clean = remove_junctions(skeletonize(binary))
    lengths = np.bincount(label(clean).ravel())[1:]
    hist, _ = np.histogram(lengths, bins=np.linspace(0, 260, 20))
    return hist + 1e-6


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b)))


# ---------------------------------------------------------------- helpers
def to_bits(chunk_f16: np.ndarray) -> np.ndarray:
    c = np.clip(chunk_f16, 0, 1).astype(np.float16)
    c[c == 0] = 0  # fold -0.0 into +0.0
    return c.view(np.uint16)


def counts_above(hist: np.ndarray, t: float) -> float:
    """Number of scores strictly greater than t."""
    idx = np.searchsorted(BIN_VALUES, t, side="right")
    return float(hist[idx:].sum())


def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {"prec": p, "rec": r, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


def pooled_at(pos_hist, neg_hist, t):
    tp = counts_above(pos_hist, t)
    fp = counts_above(neg_hist, t)
    fn = float(pos_hist.sum()) - tp
    return scores(tp, fp, fn)


def pr_curve_and_ap(pos_hist, neg_hist):
    """Exact PR curve over the distinct float16 scores, descending; AP as in sklearn."""
    nz = np.nonzero(pos_hist + neg_hist)[0][::-1]
    tp = np.cumsum(pos_hist[nz]).astype(np.float64)
    fp = np.cumsum(neg_hist[nz]).astype(np.float64)
    n_pos = float(pos_hist.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return BIN_VALUES[nz], precision, recall, ap


def per_image_mean(tp, fp, fn):
    """tp/fp/fn: arrays over images. Mean over images where each metric is defined."""
    out = {}
    for name, num, den in (("prec", tp, tp + fp), ("rec", tp, tp + fn),
                           ("f1", 2 * tp, 2 * tp + fp + fn), ("iou", tp, tp + fp + fn)):
        ok = den > 0
        out[name] = float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")
        out[f"n_{name}"] = int(ok.sum())
    return out


# ---------------------------------------------------------------- main passes
def histogram_pass(pred, targets, site, n_sites):
    """Pooled histograms per (variant, site), soft-target SSE, per-image SSIM."""
    hists = {v: {"pos": np.zeros((n_sites, NBINS), np.int64), "neg": np.zeros((n_sites, NBINS), np.int64)}
             for v in VARIANTS}
    sse = np.zeros(n_sites)
    npx = np.zeros(n_sites)
    ssim = np.zeros(len(pred))
    for i0 in range(0, len(pred), CHUNK):
        p16 = np.asarray(pred[i0:i0 + CHUNK])
        bits = to_bits(p16)
        s = site[i0:i0 + CHUNK]
        soft = np.asarray(targets["dilated"][i0:i0 + CHUNK]).astype(np.float32) / 255.0
        pf = p16.astype(np.float32)
        for k in range(n_sites):
            m = s == k
            if not m.any():
                continue
            sse[k] += float(((pf[m] - soft[m]) ** 2).sum())
            npx[k] += pf[m].size
            for v in VARIANTS:
                y = np.asarray(targets[v][i0:i0 + CHUNK])[m] > 0
                b = bits[m]
                hists[v]["pos"][k] += np.bincount(b[y], minlength=NBINS)[:NBINS]
                hists[v]["neg"][k] += np.bincount(b[~y], minlength=NBINS)[:NBINS]
        ssim[i0:i0 + len(pf)] = structural_similarity_index_measure(
            torch.from_numpy(pf)[:, None], torch.from_numpy(soft)[:, None],
            data_range=1.0, reduction="none").numpy()
    return hists, sse, npx, ssim


def per_image_pass(pred, targets, thresholds_by_variant, fracsim_by_variant):
    """Per-image confusion counts at the given thresholds, and FracSim."""
    n = len(pred)
    conf = {v: {t: np.zeros((3, n)) for t in ts} for v, ts in thresholds_by_variant.items()}
    fsim = {v: {t: np.zeros(n) for t in ts} for v, ts in fracsim_by_variant.items()}
    for i0 in range(0, n, CHUNK):
        pf = np.asarray(pred[i0:i0 + CHUNK]).astype(np.float32)
        for v in set(thresholds_by_variant) | set(fracsim_by_variant):
            y = np.asarray(targets[v][i0:i0 + CHUNK]) > 0
            for t in thresholds_by_variant.get(v, ()):
                yhat = pf > t
                c = conf[v][t]
                c[0, i0:i0 + len(pf)] = (yhat & y).sum(axis=(1, 2))
                c[1, i0:i0 + len(pf)] = (yhat & ~y).sum(axis=(1, 2))
                c[2, i0:i0 + len(pf)] = (~yhat & y).sum(axis=(1, 2))
            if v in fracsim_by_variant:
                true_h = [segment_hist(y[j]) for j in range(len(pf))]
                for t in fracsim_by_variant[v]:
                    fsim[v][t][i0:i0 + len(pf)] = [chi2(segment_hist(pf[j] > t), true_h[j]) for j in range(len(pf))]
    return conf, fsim


def summarise_split(pred, targets, site, sites, thresholds_by_variant=None, fracsim_by_variant=None,
                    pr_out=None):
    n_sites = len(sites)
    hists, sse, npx, ssim = histogram_pass(pred, targets, site, n_sites)
    groups = {"all": np.arange(n_sites)} | {s: np.array([k]) for k, s in enumerate(sites)}
    res = {"regression": {}, "pooled": {}, "ap": {}, "grid_f1": {}}
    for g, ks in groups.items():
        mse = sse[ks].sum() / npx[ks].sum()
        img = np.isin(site, ks)
        res["regression"][g] = {"mse": float(mse), "psnr": float(10 * np.log10(1.0 / mse)),
                                "ssim": float(ssim[img].mean()), "n_images": int(img.sum())}
    for v in VARIANTS:
        res["pooled"][v], res["ap"][v] = {}, {}
        for g, ks in groups.items():
            pos, neg = hists[v]["pos"][ks].sum(0), hists[v]["neg"][ks].sum(0)
            ts = sorted(set(FIXED_THRESHOLDS) | set((thresholds_by_variant or {}).get(v, ())))
            res["pooled"][v][g] = {f"{t:g}": pooled_at(pos, neg, t) for t in ts}
            thr, prec, rec, ap = pr_curve_and_ap(pos, neg)
            res["ap"][v][g] = ap
            res["pooled"][v][g]["positive_pixel_frac"] = float(pos.sum() / (pos.sum() + neg.sum()))
            if g == "all":
                res["grid_f1"][v] = {f"{t:g}": pooled_at(pos, neg, t)["f1"] for t in GRID}
                if pr_out is not None:
                    keep = np.unique(np.linspace(0, len(thr) - 1, min(len(thr), 2000)).astype(int))
                    pr_out[v] = np.stack([thr[keep], prec[keep], rec[keep]], 1)
    if thresholds_by_variant or fracsim_by_variant:
        conf, fsim = per_image_pass(pred, targets, thresholds_by_variant or {}, fracsim_by_variant or {})
        res["per_image"], res["fracsim"] = {}, {}
        for v, by_t in conf.items():
            res["per_image"][v] = {g: {f"{t:g}": per_image_mean(*(c[:, np.isin(site, ks)]))
                                       for t, c in by_t.items()} for g, ks in groups.items()}
        for v, by_t in fsim.items():
            res["fracsim"][v] = {g: {f"{t:g}": float(f[np.isin(site, ks)].mean()) for t, f in by_t.items()}
                                 for g, ks in groups.items()}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("--targets", type=Path, default=Path("review_out/targets"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path("review_out/metrics") / f"{args.pred_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    info = json.loads((args.targets / "targets_info.json").read_text())
    sites = info["sites"]
    meta = json.loads((args.pred_dir / "meta.json").read_text())

    def load(split):
        pred = np.load(args.pred_dir / f"{split}_pred.npy", mmap_mode="r")
        tg = {v: np.load(args.targets / f"{split}_{v}.npy", mmap_mode="r") for v in VARIANTS}
        site = np.load(args.targets / f"{split}_site.npy")
        assert len(pred) == len(site) == len(tg["dilated"]), (split, len(pred), len(site))
        return pred, tg, site

    # 1. validation: choose one threshold per target variant (max pooled F1)
    val = summarise_split(*load("val"), sites)
    selected = {v: float(max(val["grid_f1"][v].items(), key=lambda kv: kv[1])[0]) for v in VARIANTS}
    val["selected_threshold"] = selected

    # 2. test at the fixed thresholds and at the validation-selected ones
    th = {v: sorted(set(FIXED_THRESHOLDS) | {selected[v]}) for v in VARIANTS}
    fs = {"dilated": sorted({0.1, selected["dilated"]}), "original": sorted({0.1, selected["original"]})}
    pr = {}
    test = summarise_split(*load("test"), sites, th, fs, pr_out=pr)

    np.savez_compressed(out.with_suffix(".pr_test.npz"), **pr)
    result = {"tag": args.pred_dir.name, "meta": meta, "sites": sites, "selected_threshold": selected,
              "val": {k: val[k] for k in ("regression", "pooled", "ap", "grid_f1", "selected_threshold")},
              "test": test}
    out.write_text(json.dumps(result, indent=1))
    t = test["pooled"]["dilated"]["all"]
    sel_key = f"{selected['dilated']:g}"
    print(f"{args.pred_dir.name}: sel(dilated)={selected['dilated']}  "
          f"F1@0.1={t['0.1']['f1']:.3f} IoU@0.1={t['0.1']['iou']:.3f}  "
          f"F1@sel={t[sel_key]['f1']:.3f}  AP={test['ap']['dilated']['all']:.3f}  "
          f"MSE={test['regression']['all']['mse']:.4f}")
    print("Wrote", out)


if __name__ == "__main__":
    main()
