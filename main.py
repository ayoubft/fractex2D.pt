import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io
from sklearn.exceptions import UndefinedMetricWarning
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.train import train_loop, eval_loop, save_metrics
from src.visualize import plot_example, plot_result

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Paths to test images
img_paths = [
    # 'data/kl5-s3',
    # 'data/hnn-z1-s3',
    # 'data/matteo21-z1-s3',
]


@hydra.main(config_name="config.yaml", config_path="config", version_base=None)
def main(cfg: DictConfig):
    """Train and evaluate model, generate plots, and predict test images."""

    # --- 1. Setup output directories ---
    print("Config:\n", OmegaConf.to_yaml(cfg))
    save_path = Path(HydraConfig.get().runtime.output_dir)
    (save_path / 'samples').mkdir(exist_ok=True)
    (save_path / 'predictions').mkdir(exist_ok=True)
    print("Output dir:", save_path)

    writer = SummaryWriter(save_path)

    # --- 2. Load dataset ---
    trainloader, valloader, testloader = instantiate(cfg.dataset)
    print("Data loaded!")

    for _ in range(11):
        plot_example(trainloader, save_path, cfg.in_channels)

    # --- 3. Instantiate model ---
    model = instantiate(cfg.model)
    print("Model loaded!")

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
    torch.manual_seed(cfg.seed)
    model = model.to(device)

    # --- 4. Define loss, optimizer, scheduler ---
    loss_name = cfg.loss._target_.split('.')[-1]
    if loss_name == 'BCEWithLogitsLoss':
        criterion = instantiate(cfg.loss, pos_weight=torch.Tensor([cfg.param]).to(device))
    else:
        criterion = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)

    valid_loss_min = np.inf

    # --- 5. Train model ---
    for epoch in range(cfg.epochs):
        print(f"Starting epoch {epoch}")
        train_loss = train_loop(model, optimizer, criterion, trainloader, device, cfg.model._target_.split('.')[-1])
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Validation
        model.eval()
        running_vloss = 0
        with torch.no_grad():
            for imgs, labels in tqdm(valloader, desc="Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                running_vloss += criterion(out, labels).item() * imgs.shape[0]
        running_vloss /= len(valloader.sampler)
        writer.add_scalars("Training vs. Validation Loss",
                           {"Training": train_loss, "Validation": running_vloss}, epoch + 1)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, vloss={running_vloss:.4f}")

        # Save best model
        if running_vloss <= valid_loss_min:
            torch.save(model.state_dict(), save_path / "model.pt")
            valid_loss_min = running_vloss
            print(f"Saved best model at epoch {epoch}")

    # Save final model
    torch.save(model.state_dict(), save_path / "final_model.pt")

    # --- 6. Load best model ---
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(save_path / "model.pt", map_location=device))
    model = model.to(device)

    # --- 7. Plot test samples ---
    for _ in range(17):
        plot_result(model, testloader, save_path, cfg.dataset.shape, cfg.in_channels, device)

    # --- 8. Evaluate on test set ---
    test_metrics = eval_loop(model, scheduler, criterion, testloader,
                             cfg.threshold, device,
                             cfg.model._target_.split('.')[-1],
                             cfg.ignore_index)
    save_metrics(test_metrics, 'test', writer, cfg.epochs - 1)
    print(test_metrics)

    # --- 9. Predict test images ---
    model.eval()
    for img_path in img_paths:
        img = io.imread(f"{img_path}.png")
        dem = io.imread(f"{img_path}-dem.tif")
        dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
        combined = np.concatenate((img[:, :, :3], np.expand_dims(dem, 2)), axis=2)

        patch_size = cfg.dataset.shape
        h, w, c = combined.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        combined_padded = np.pad(combined, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        patches = patchify(combined_padded, (patch_size, patch_size, cfg.in_channels), step=256)

        pred_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = torch.Tensor(patches[i, j]).permute(0, 3, 1, 2) / 255.0
                with torch.no_grad():
                    pred_patches.append(model(patch.to(device)).cpu())

        pred = np.reshape(pred_patches,
                          (patches.shape[0], patches.shape[1], 1, patch_size, patch_size, 1))
        pred = unpatchify(pred, combined_padded.shape[:2] + (1,))
        pred = pred[:h, :w, :]
        pred_img = Image.fromarray((pred.reshape(img.shape[0], img.shape[1]) * 255).astype(np.uint8))

        pred_img.save(save_path / f"pred_proba_{Path(img_path).name}.png")


if __name__ == "__main__":
    main()
