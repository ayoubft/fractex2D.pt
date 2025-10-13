import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from patchify import patchify, unpatchify
from datetime import datetime
from PIL import Image
from skimage import io
from sklearn.base import BaseEstimator
from torchmetrics.classification import BinaryF1Score
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm

from src.dataset_benchm import MATTEO, OVAS, SAMSU
from torch.utils.data import ConcatDataset, DataLoader, Subset


model_path = '/users/afatihi/work-detect/fractex2D.pt/multirun_BM_/segformer/2025-08-30_16-50/model=sm_segformer'


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    save_path = model_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'),
                          weights_only=True, map_location=torch.device('cpu')))
    # , map_location=torch.device('cpu')  ## torch.load
    model.to(device)
    CSV_FILE = f'{save_path}/f1_{timestamp}.csv'
    IGNORE_INDEX = None

    # class Estimator(BaseEstimator):
    #     def __init__(self, threshold=.5):
    #         self.threshold = threshold

    #     def fit(self, X, y=None):
    #         print('Fitting starts now ...')
    #         testloader = DataLoader(X, batch_size=1)
    #         self.model = model
    #         model.eval()
    #         with torch.no_grad():
    #             f1_metric = BinaryF1Score(ignore_index=IGNORE_INDEX).to(device)
    #             pbar = tqdm(testloader, desc='Iterating over test data', disable=True)
    #             for imgs, labels in pbar:
    #                 imgs = imgs.to(device)
    #                 labels = labels.to(device)
    #                 out = model.to(device)(imgs)
    #                 predicted_clf = (out > self.threshold).float()
    #                 labels_clf = (labels > 0.).float()
    #                 f1_metric(predicted_clf, labels_clf)

    #         f1 = f1_metric.compute().item()

    #         self.f1 = f1
    #         print(f'Thr: {self.threshold}, F1: {self.f1}')

    #         return self

    #     def score(self, X, y=None):
    #         row = {
    #             'threshold': self.threshold,
    #             'f1': self.f1
    #         }

    #         df = pd.DataFrame([row])
    #         if not os.path.exists(CSV_FILE):
    #             df.to_csv(CSV_FILE, index=False)
    #         else:
    #             df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    #         return self.f1

    # param_grid = {
    #     'threshold': np.arange(0., .9, .05),
    # }

    datasets = {
        'ovaskainen23': OVAS,
        'samsu19': SAMSU,
        'matteo21': MATTEO,
    }

    # all_ds = []
    # # for ds in [OVAS, SAMSU, MATTEO]:
    # print(cfg.dataset.datasets)
    # for ds in [datasets[cfg.dataset.datasets]]:
    #     testset_ = ds(subset='test', expand=True, dilate=True,
    #                   in_channels=cfg.in_channels)
    #     # all_ds.append(Subset(testset_, np.arange(SIZE)))
    #     all_ds.append(testset_)

    # all_ds = ConcatDataset(all_ds)
    # print('Data loaded!')
    # print('Starting estimating!')
    # grid = GridSearchCV(Estimator(), param_grid, scoring=None, cv=2)
    # grid.fit(all_ds)

    # print(grid.best_params_)
    # print(grid.best_score_)

    param_grid = np.arange(0., .9, .05)

    all_ds = []
    print(cfg.dataset.datasets)
    for ds in [OVAS, SAMSU, MATTEO]:
    # for ds in [datasets[cfg.dataset.datasets]]:
        testset_ = ds(subset='test', expand=True, dilate=True,
                      in_channels=cfg.in_channels)
        all_ds.append(testset_)
    all_ds = ConcatDataset(all_ds)
    testloader = DataLoader(all_ds, batch_size=1)

    model.eval()
    ods_scores = []
    with torch.no_grad():
        for thr in param_grid:
            f1_metric = BinaryF1Score(ignore_index=IGNORE_INDEX).to(device)
            for img, label in tqdm(testloader, desc="ODS"):
                img, label = img.to(device), label.to(device)
                out = model(img)
                predicted = (out > thr).float()
                labels_clf = (label > 0.).float()
                f1_metric(predicted, labels_clf)
            ods_scores.append(f1_metric.compute().item())

    ods = max(ods_scores)
    best_thr = param_grid[int(np.argmax(ods_scores))]
    print(f"ODS: {ods} at threshold {best_thr}")


if __name__ == "__main__":
    main()
