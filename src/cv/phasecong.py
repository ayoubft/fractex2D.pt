import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from phasepack import phasecong
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.dataset_benchm import MATTEO, OVAS, SAMSU
from src.train import eval_loop
from src.visualize import plot_result

datasets = {
    'ovaskainen23': OVAS,
    'samsu19': SAMSU,
    'matteo21': MATTEO,
}


for ds_nm in datasets.keys():

    ALGO = 'phasecong'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILE = f'CV_outputs/grid_metrics_{ALGO}_{ds_nm}_{timestamp}.csv'
    PLOT_DIR = f'CV_predictions/{ALGO}_{ds_nm}_{timestamp}'
    N = 99
    in_ch = 4 if ds_nm == 'matteo21' else 3
    os.makedirs(PLOT_DIR+'/predictions', exist_ok=True)
    N_RANDOM = 50
    np.random.seed(42)

    print(ALGO)
    print(timestamp)
    print(ds_nm)

    EXPAND, DILATE = True, True
    all_ds = []
    for ds in [datasets[ds_nm]]:
        testset_ = ds(subset='test', expand=EXPAND, dilate=DILATE, in_channels=4)
        all_ds.append(Subset(testset_, np.arange(N)))
        # all_ds.append(testset_)

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
        """Edge detection model using Canny on PCA-reduced channels."""
        def __init__(self, t=0.5, nscale=5, norient=6, minWaveLength=3, mult=2.1,
                     sigmaOnf=0.55, k=2.0, cutOff=0.5, g=10.0, noiseMethod=-1,
                     channel_idx=0):
            super(ThresholdModel, self).__init__()
            self.t = float(t)
            self.nscale = int(nscale)
            self.norient = int(norient)
            self.minWaveLength = float(minWaveLength)
            self.mult = float(mult)
            self.sigmaOnf = float(sigmaOnf)
            self.k = float(k)
            self.cutOff = float(cutOff)
            self.g = float(g)
            self.noiseMethod = int(noiseMethod)
            self.channel_idx = int(channel_idx)

        def forward(self, x):
            if isinstance(x, (list, tuple)):
                x = x[0]
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
                # img = (img - img.mean()) / (img.std() + 1e-8)
                M, _, _, _, _, _, _ = phasecong(
                    img1,
                    nscale=self.nscale,
                    norient=self.norient,
                    minWaveLength=self.minWaveLength,
                    mult=self.mult,
                    sigmaOnf=self.sigmaOnf,
                    k=self.k,
                    cutOff=self.cutOff,
                    g=self.g,
                    noiseMethod=self.noiseMethod,
                )
                mask = (M > self.t).astype(np.uint8)
                out_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                outs.append(out_t)
            return torch.cat(outs, dim=0)

    CALCULATED_METRICS = ('mse', 'psnr', 'ssim', 'ae', 'acc', 'f1',
                          'prec', 'rec', 'roc_auc', 'iou_nbg', 'loss')

    class Estimator(BaseEstimator):
        """Sklearn-compatible estimator for grid search optimization."""
        def __init__(self, threshold=.5, nscale=5, norient=6, minWaveLength=3,
                     mult=2.1, sigmaOnf=0.55, k=2.0, cutOff=0.5,
                     g=10.0, noiseMethod=-1):
            self.threshold = threshold
            self.nscale = nscale
            self.norient = norient
            self.minWaveLength = minWaveLength
            self.mult = mult
            self.sigmaOnf = sigmaOnf
            self.k = k
            self.cutOff = cutOff
            self.g = g
            self.noiseMethod = noiseMethod

        def fit(self, X, y=None):
            dataset = X
            valloader = DataLoader(dataset, batch_size=1)
            self.model = ThresholdModel(
                t=self.threshold,
                nscale=self.nscale,
                norient=self.norient,
                minWaveLength=self.minWaveLength,
                mult=self.mult,
                sigmaOnf=self.sigmaOnf,
                k=self.k,
                cutOff=self.cutOff,
                g=self.g,
                noiseMethod=self.noiseMethod).cpu()
            criterion = nn.HuberLoss()
            self.model.eval()
            scheduler = None
            self.metrics = eval_loop(self.model, scheduler, criterion,
                                     valloader, self.threshold, 'cpu')
            self.val_accuracy = -self.metrics['loss']
            print(f'Val accuracy (loss): {self.val_accuracy}')
            return self

        def score(self, X, y=None):
            row = {
                'threshold': self.threshold,
                'nscale': self.nscale,
                'norient': self.norient,
                'minWaveLength': self.minWaveLength,
                'mult': self.mult,
                'sigmaOnf': self.sigmaOnf,
                'k': self.k,
                'cutOff': self.cutOff,
                'g': self.g,
                'noiseMethod': self.noiseMethod,
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
        'nscale': [3, 4, 5],
        'norient': [4, 6],
        'minWaveLength': [3, 4, 5],
        'mult': [1.8, 2.1, 2.4],
        'sigmaOnf': [0.45, 0.55, 0.65],
        'k': [2.0, 3.0],
        'cutOff': [0.35, 0.5, 0.65],
        'g': [10.0, 20.0],
        'noiseMethod': [-1],
    }

    grid = GridSearchCV(Estimator(), param_grid, scoring=None, cv=2, n_jobs=1)
    grid.fit(all_ds)

    print(grid.best_params_)
    print(grid.best_score_)

    bp = grid.best_params_
    model = ThresholdModel(
        t=bp.get('threshold', 0.5),
        nscale=bp.get('nscale', 5),
        norient=bp.get('norient', 6),
        minWaveLength=bp.get('minWaveLength', 3),
        mult=bp.get('mult', 2.1),
        sigmaOnf=bp.get('sigmaOnf', 0.55),
        k=bp.get('k', 2.0),
        cutOff=bp.get('cutOff', 0.5),
        g=bp.get('g', 10.0),
        noiseMethod=bp.get('noiseMethod', -1)
    ).cpu()

    # indices = np.random.choice(
    #     len(all_ds), size=min(N_RANDOM, len(all_ds)), replace=False)

    # for idx in indices:
    #     subset = Subset(all_ds, [int(idx)])
    #     dl = DataLoader(subset, batch_size=1)
    #     plot_result(model, dl, PLOT_DIR, 256, in_ch, 'cpu')

    df = pd.read_csv(CSV_FILE)

    with open(f'CV_outputs/{ALGO}_{ds_nm}_{timestamp}.json', 'w') as fp:
        json.dump(grid.best_params_, fp)
