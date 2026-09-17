import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io
from tqdm.auto import tqdm

from src.train2 import eval_loop

# Path to the trained model
model_path = (
    "/users/afatihi/work-detect/fractex2D.pt/multirun_BM_/unetlogit/2026-01-29_09-54/model=unet_logit"
)

# Test image paths
img_paths = [
    "data/test_ovas/kl5/kl5-s3",
    "data/test_ovas/kl5/hnn-z1-s3",
    "data/test_ovas/kl5/_matteo21-z1-s3",
    "data/test_ovas/kl5/ortho-ldb-z1-s3",
]


@hydra.main(
    config_name="config.yaml",
    config_path=os.path.join(model_path, ".hydra"),
    version_base=None,
)
def main(cfg: DictConfig):
    """Load trained model, evaluate test set, and generate patch-based predictions."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"
    )
    print("Using device:", device)

    save_path = Path(model_path)
    model_name = cfg.model._target_.split(".")[-1]
    print("Model:", model_name)

    # Load model and related objects
    model = instantiate(cfg.model).to(device)
    model.load_state_dict(
        torch.load(save_path / "final_model.pt", map_location="cpu")
    )
    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)
    criterion = instantiate(cfg.loss)

    trainloader, valloader, testloader = instantiate(cfg.dataset)

    # Adaptive BatchNorm on test set
    for _ in range(0):
        model.train()
        for X_t, _ in tqdm(testloader, desc="Adaptive BN"):
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.track_running_stats = True
            _ = model(X_t.to(device))

    torch.cuda.empty_cache()
    model.eval()

    # Evaluate on test set
    test_metrics = eval_loop(model, scheduler, criterion, testloader,
                             cfg.threshold, device, model_name)
    print(cfg.dataset.datasets)
    print("Test metrics:", test_metrics)

    # Generate predictions for test images
    patch_size = cfg.dataset.shape
    for img_path in img_paths:
        img = io.imread(f"{img_path}.png")
        dem = io.imread(f"{img_path}-dem.tif")
        dem = (dem - dem.mean()) / (dem.std() + 1e-8)
        combined = np.concatenate((img[:, :, :3], np.expand_dims(dem, 2)), axis=2)

        h, w, c = combined.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        combined_padded = np.pad(
            combined,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
        )

        # Split into patches
        patches = patchify(combined_padded, (patch_size, patch_size, c), step=patch_size)

        pred_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = torch.Tensor(patches[i, j]).permute(0, 3, 1, 2) / 255.0
                with torch.no_grad():
                    pred_patches.append(model(patch.to(device)).cpu())
                    # pred_patches.append(torch.sigmoid(model(patch.to(device))).cpu())

        # Reconstruct full prediction
        pred = np.reshape(
            pred_patches,
            (patches.shape[0], patches.shape[1], 1, patch_size, patch_size, 1),
        )
        pred = unpatchify(pred, combined_padded.shape[:2] + (1,))
        pred = pred[:h, :w, :]
        pred_img = Image.fromarray((pred.reshape(h, w) * 255).astype(np.uint8))
        pred_img.save(save_path / f"{cfg.dataset.datasets}_pred_{Path(img_path).name}.png")


if __name__ == "__main__":
    main()
