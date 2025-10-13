import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.filters import sobel
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.dataset_benchm import MATTEO, OVAS, SAMSU, dilate_labels
from src.train2 import eval_loop
from src.visualize import plot_result

ALGO = 'sobel'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f'CV_outputs/grid_metrics_{ALGO}_{timestamp}.csv'
N = 200

EXPAND, DILATE = True, True
all_ds = []
for ds in [OVAS, MATTEO, SAMSU]:
    testset_ = ds(subset='test', expand=EXPAND, dilate=DILATE)
    all_ds.append(Subset(testset_, np.arange(N)))
all_ds = ConcatDataset(all_ds)

print(ALGO)
print(timestamp)


class ThresholdModel(nn.Module):
    def __init__(self, t=0.5, channel_idx=0):
        super().__init__()
        self.t = float(t)
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
            sob_img = sobel(img)
            mask = (sob_img > self.t).astype(np.uint8)
            mask = dilate_labels(mask)
            out_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            outs.append(out_t)
        return torch.cat(outs, dim=0)


CALCULATED_METRICS = ('mse', 'psnr', 'ssim', 'ae', 'acc', 'f1',
                      'prec', 'rec', 'roc_auc', 'iou', 'loss')


class Estimator(BaseEstimator):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        dataset = X
        valloader = DataLoader(dataset, batch_size=1)
        self.model = ThresholdModel(t=self.threshold).cpu()
        criterion = nn.HuberLoss()
        self.model.eval()
        scheduler = None
        self.metrics = eval_loop(self.model, scheduler, criterion, valloader,
                                 self.threshold, 'cpu')
        self.val_accuracy = -self.metrics['loss']
        return self

    def score(self, X, y=None):
        row = {'threshold': self.threshold, **self.metrics}
        df = pd.DataFrame([row])
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        return self.val_accuracy


param_grid = {
    'threshold': np.arange(0.0, 1, 0.1),
}

grid = GridSearchCV(Estimator(), param_grid, scoring=None, cv=2, n_jobs=1)
grid.fit(all_ds)

print(grid.best_params_)
print(grid.best_score_)

bp = grid.best_params_
model = ThresholdModel(t=bp.get('threshold', 0.5)).cpu()

PLOT_DIR = f'plots/{ALGO}_{timestamp}'
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
