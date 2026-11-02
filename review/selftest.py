"""Check review.evaluate against reference implementations on synthetic data.

    python -m review.selftest

* pooled P/R/F1/IoU vs torchmetrics Binary* (what src/train2.eval_loop uses)
* AP vs sklearn.metrics.average_precision_score
* vectorised remove_junctions / FracSim vs src.train2 originals
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review import evaluate as ev  # noqa: E402
from src.train2 import fracture_similarity, remove_junctions  # noqa: E402

rng = np.random.default_rng(0)
n, h = 12, 64
y = np.zeros((n, h, h), bool)
for i in range(n):
    for _ in range(4):
        r = rng.integers(0, h)
        y[i, r, :] = True
        y[i, :, rng.integers(0, h)] = True
p = np.clip(0.6 * y + rng.normal(0.15, 0.2, y.shape), 0, 1).astype(np.float16)

bits = ev.to_bits(p)
pos = np.bincount(bits[y], minlength=ev.NBINS)
neg = np.bincount(bits[~y], minlength=ev.NBINS)

for t in (0.05, 0.1, 0.37, 0.5):
    mine = ev.pooled_at(pos, neg, t)
    pt, yt = torch.from_numpy((p.astype(np.float32) > t).astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    ref = {k: float(m(pt, yt)) for k, m in (("prec", BinaryPrecision()), ("rec", BinaryRecall()),
                                             ("f1", BinaryF1Score()), ("iou", BinaryJaccardIndex()))}
    for k in ref:
        assert abs(mine[k] - ref[k]) < 1e-6, (t, k, mine[k], ref[k])
print("pooled metrics == torchmetrics")

_, _, _, ap = ev.pr_curve_and_ap(pos, neg)
ap_ref = average_precision_score(y.ravel(), p.astype(np.float32).ravel())
assert abs(ap - ap_ref) < 1e-9, (ap, ap_ref)
print(f"AP == sklearn ({ap:.6f})")

for i in range(n):
    sk = rng.random((h, h)) > 0.7
    assert np.array_equal(ev.remove_junctions(sk), remove_junctions(sk)), i
    a, b = ev.chi2(ev.segment_hist(p[i] > 0.3), ev.segment_hist(y[i])), fracture_similarity(
        torch.from_numpy((p[i].astype(np.float32) > 0.3).astype(np.float32)), torch.from_numpy(y[i].astype(np.float32)))
    assert abs(a - b) < 1e-9, (i, a, b)
print("remove_junctions / FracSim == src.train2")
print("ALL OK")
