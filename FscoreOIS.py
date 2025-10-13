import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io
from datetime import datetime
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
    IGNORE_INDEX = None

    param_grid = np.arange(0., .9, .05)

    datasets = {
        'ovaskainen23': OVAS,
        'samsu19': SAMSU,
        'matteo21': MATTEO,
    }

    all_ds = []
    print(cfg.dataset.datasets)
    for ds in [OVAS, SAMSU, MATTEO]:
    # for ds in [datasets[cfg.dataset.datasets]]:
        testset_ = ds(subset='test', expand=True, dilate=True,
                      in_channels=cfg.in_channels)
        all_ds.append(testset_)
    all_ds = ConcatDataset(all_ds)
    print('Data loaded!')
    print('Starting estimating!')

    ois_scores = []
    testloader = DataLoader(all_ds, batch_size=1)

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(testloader, desc="Per-image OIS"):
            img, label = img.to(device), label.to(device)
            best_f1 = 0.0
            for thr in param_grid:
                out = model(img)
                predicted = (out > thr).float()
                labels_clf = (label > 0.).float()
                f1_metric = BinaryF1Score(ignore_index=IGNORE_INDEX).to(device)
                f1_metric(predicted, labels_clf)
                f1 = f1_metric.compute().item()
                best_f1 = max(best_f1, f1)
            ois_scores.append(best_f1)

    mean_ois = np.mean(ois_scores)
    print(f"OIS: {mean_ois}")


if __name__ == "__main__":
    main()
