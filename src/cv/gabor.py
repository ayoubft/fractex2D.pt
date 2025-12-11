import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.filters import gabor
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.dataset_benchm import MATTEO, OVAS, SAMSU, dilate_labels
from src.train import eval_loop
from src.visualize import plot_result

datasets = {
    'ovaskainen23': OVAS,
    'samsu19': SAMSU,
    'matteo21': MATTEO,
}

for ds_nm in datasets.keys():
    ALGO = 'gabor'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILE = f'CV_outputs/grid_metrics_{ALGO}_{ds_nm}_{timestamp}.csv'
    N = 99
    in_ch = 4 if ds_nm == 'matteo21' else 3

    print(ALGO)
    print(timestamp)
    print(ds_nm)

    EXPAND, DILATE = True, True
    all_ds = []
    for ds in [datasets[ds_nm]]:
        testset_ = ds(subset='test', expand=EXPAND, dilate=DILATE, in_channels=4)
        # all_ds.append(Subset(testset_, np.arange(N)))
        all_ds.append(testset_)
    all_ds = ConcatDataset(all_ds)
    print('Data loaded')

    xs = []
    for i in range(len(all_ds)):
        x, _ = all_ds[i]
        if x.shape[0] == 4:
            arr = x.numpy().astype(np.float64)
            arr = (arr - arr.mean(axis=(1, 2), keepdims=True)) / (arr.std(axis=(1, 2), keepdims=True) + 1e-8)
            xs.append(arr.reshape(4, -1).T)

    if xs:
        X = np.concatenate(xs, axis=0)
        pc = PCA(n_components=1)
        pc.fit(X)

    print('PCA fitted')

    class ThresholdModel(nn.Module):
        def __init__(self, t=0.5, frequency=0.2, channel_idx=0, debug=False):
            super().__init__()
            self.t = float(t)
            self.frequency = float(frequency)
            self.channel_idx = int(channel_idx)
            self.debug = debug

        def forward(self, x):
            if isinstance(x, (list, tuple)):
                x = x[0]
            assert torch.is_tensor(x), "forward expects a torch.Tensor"
            x = x.detach().cpu()
            B, C, H, W = x.shape
            outs = []
            for i in range(B):
                img = x.numpy().astype(np.float64)
                arr = (img - img.mean(axis=(1, 2), keepdims=True)) / (img.std(axis=(1, 2), keepdims=True) + 1e-8)
                flat = arr.reshape(4, -1).T
                comp = pc.transform(flat)
                img1 = comp.T.reshape(H, W)
                # img = x[i, self.channel_idx, :, :].numpy().astype(np.float64)
                # if img.size == 0:
                #     img = x[i, 0, :, :].numpy().astype(np.float64)
                # img = (img - img.mean()) / (img.std() + 1e-8)
                real_fil, imag_fil = gabor(img1, frequency=self.frequency)
                mask = (real_fil > self.t).astype(np.uint8)
                mask = dilate_labels(mask)
                out_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                outs.append(out_t)
            out = torch.cat(outs, dim=0)
            return out

    CALCULATED_METRICS = ('mse', 'psnr', 'ssim', 'ae', 'acc', 'f1',
                          'prec', 'rec', 'roc_auc', 'iou_nbg', 'loss')

    class Estimator(BaseEstimator):
        """Sklearn-compatible estimator for grid search optimization."""
        def __init__(self, threshold=.5, frequency=.7):
            self.threshold = threshold
            self.frequency = frequency

        def fit(self, X, y=None):
            dataset = X
            valloader = DataLoader(dataset, batch_size=1)
            self.model = ThresholdModel(t=self.threshold,
                                        frequency=self.frequency).cpu()
            criterion = nn.HuberLoss()
            self.model.eval()
            scheduler = None
            self.metrics = eval_loop(self.model, scheduler, criterion,
                                     valloader, self.threshold, 'cpu')
            self.val_accuracy = -self.metrics['loss']
            return self

        def score(self, X, y=None):
            row = {
                'threshold': self.threshold,
                'frequency': self.frequency,
                **self.metrics
            }
            df = pd.DataFrame([row])
            if not os.path.exists(CSV_FILE):
                df.to_csv(CSV_FILE, index=False)
            else:
                df.to_csv(CSV_FILE, mode='a', header=False, index=False)
            return self.val_accuracy

    param_grid = {
        'threshold': [0.1, 0.3, 0.5, 0.7, 0.9],
        'frequency': [0.05, 0.25, 0.45, 0.7, 0.9],
    }

    grid = GridSearchCV(Estimator(), param_grid, scoring=None,
                        cv=2, n_jobs=1)
    grid.fit(all_ds)

    print(grid.best_params_)
    print(grid.best_score_)

    bp = grid.best_params_
    model = ThresholdModel(t=bp['threshold'], frequency=bp['frequency']).cpu()

    # PLOT_DIR = f'plots/{ALGO}_{ds_nm}_{timestamp}'
    # os.makedirs(PLOT_DIR+'/predictions', exist_ok=True)

    # testloader = DataLoader(all_ds, batch_size=10)
    # for _ in range(9):
    #     plot_result(model, testloader, PLOT_DIR, 256, in_ch, 'cpu')

    # N_RANDOM = 100
    # np.random.seed(42)
    # indices = np.random.choice(
    #     len(all_ds), size=min(N_RANDOM, len(all_ds)), replace=False)
    # for idx in indices:
    #     subset = Subset(all_ds, [int(idx)])
    #     dl = DataLoader(subset, batch_size=1)
    #     plot_result(model, dl, PLOT_DIR, 256, in_ch, 'cpu')

    with open(f'CV_outputs/{ALGO}_{ds_nm}_{timestamp}.json', 'w') as fp:
        json.dump(grid.best_params_, fp)
