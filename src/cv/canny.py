import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.feature import canny
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.dataset_benchm import MATTEO, OVAS, SAMSU, dilate_labels
from src.train2 import eval_loop
from src.visualize import plot_result

ALGO = 'canny'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f'CV_outputs_3/grid_metrics_{ALGO}_{timestamp}.csv'
N = 200

EXPAND, DILATE = True, True
all_ds = []
for ds in [OVAS, MATTEO, SAMSU]:
    testset_ = ds(subset='test', expand=EXPAND, dilate=DILATE, transform=False)
    # all_ds.append(Subset(testset_, np.arange(N)))
    all_ds.append(testset_)
all_ds = ConcatDataset(all_ds)


class ThresholdModel(nn.Module):
    def __init__(self, t=0.5, sigma=1.0,
                 low_threshold=0.1, high_threshold=0.2, channel_idx=0):
        super().__init__()
        self.t = float(t)
        self.sigma = float(sigma)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.channel_idx = int(channel_idx)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        assert torch.is_tensor(x), "forward expects a torch.Tensor"
        x = x.detach().cpu()
        B, C, H, W = x.shape
        outs = []
        for i in range(B):
            img = x[i, self.channel_idx, :, :].numpy().astype(np.float64)
            edges = canny(
                img, sigma=self.sigma, low_threshold=self.low_threshold,
                high_threshold=self.high_threshold)
            mask = edges.astype(np.uint8)
            mask = dilate_labels(mask)
            out_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            outs.append(out_t)
        return torch.cat(outs, dim=0)


CALCULATED_METRICS = ('mse', 'psnr', 'ssim', 'ae', 'acc', 'f1',
                      'prec', 'rec', 'roc_auc', 'iou', 'loss')


class Estimator(BaseEstimator):
    def __init__(self, threshold=.5, sigma=1.0,
                 low_threshold=.1, high_threshold=.2):
        self.threshold = threshold
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def fit(self, X, y=None):
        dataset = X
        valloader = DataLoader(dataset, batch_size=1)
        self.model = ThresholdModel(t=self.threshold, sigma=self.sigma,
                                    low_threshold=self.low_threshold,
                                    high_threshold=self.high_threshold).cpu()
        criterion = nn.HuberLoss()
        self.model.eval()
        scheduler = None
        self.metrics = eval_loop(self.model, scheduler, criterion, valloader,
                                 self.threshold, 'cpu')
        self.val_accuracy = -self.metrics['loss']
        return self

    def score(self, X, y=None):
        row = {
            'threshold': self.threshold,
            'sigma': self.sigma,
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold,
            **self.metrics
        }
        df = pd.DataFrame([row])
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        return self.val_accuracy


param_grid = {
    'threshold': np.linspace(0.05, 0.95, 5),
    'sigma': [0.5, 1.0, 1.5, 2],
    'low_threshold': [0.1, 0.15, 0.29],
    'high_threshold': [0.3, 0.4, 0.5],
}

grid = RandomizedSearchCV(Estimator(), param_grid,
                          scoring=None, cv=2,
                          n_jobs=1, n_iter=20)
grid.fit(all_ds)

print(grid.best_params_)
print(grid.best_score_)

bp = grid.best_params_
model = ThresholdModel(t=bp.get('threshold', 0.5),
                       sigma=bp.get('sigma', 1.0),
                       low_threshold=bp.get('low_threshold', 0.1),
                       high_threshold=bp.get('high_threshold', 0.2)).cpu()

PLOT_DIR = f'CV_outputs_3/plots/{ALGO}_{timestamp}'
os.makedirs(PLOT_DIR+'/predictions', exist_ok=True)

testloader = DataLoader(all_ds, batch_size=10)
for _ in range(9):
    plot_result(model, testloader, PLOT_DIR, 256, 3, 'cpu')

N_RANDOM = 100
np.random.seed(42)
indices = np.random.choice(
    len(all_ds), size=min(N_RANDOM, len(all_ds)), replace=False)
for idx in indices:
    subset = Subset(all_ds, [int(idx)])
    dl = DataLoader(subset, batch_size=1)
    plot_result(model, dl, PLOT_DIR, 256, 3, 'cpu')

with open(f'CV_outputs/best_{ALGO}_{timestamp}.json', 'w') as fp:
    json.dump(grid.best_params_, fp)
