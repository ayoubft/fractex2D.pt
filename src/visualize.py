import os
import matplotlib.pyplot as plt
import numpy as np


def plot_example(loader, save_path, in_channels):
    """
    Plot a random batch example from the loader and save to disk.

    Args:
        loader: PyTorch DataLoader providing images and masks.
        save_path: Directory where the figure will be saved.
        in_channels: Number of input channels in the image.
    """
    fig, axes = plt.subplots(
        1, max(2, in_channels - 1), figsize=(4 + in_channels * 2, 5)
    )

    images, masks = next(iter(loader))
    ind = np.random.choice(loader.batch_size)

    image = images[ind].permute(1, 2, 0)[:, :, :min(in_channels, 3)]
    mask = masks[ind].permute(1, 2, 0)

    axes[0].imshow(image.numpy())
    axes[0].set_title("Image")
    axes[0].axis("off")

    if in_channels == 4:
        dem = images[ind].permute(1, 2, 0)[:, :, 3]
        axes[1].imshow(dem.numpy(), cmap="gray")
        axes[1].set_title("DEM")
        axes[1].axis("off")

    axes[in_channels - 2].imshow(mask.numpy(), cmap="gray")
    axes[in_channels - 2].set_title("Mask")
    axes[in_channels - 2].axis("off")

    fig.tight_layout()
    save_dir = os.path.join(save_path, "samples")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{ind}.png"))
    plt.close(fig)


def plot_result(model, loader, save_path, shape, in_channels, device):
    """
    Plot model prediction vs ground truth for a random batch example.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader providing input images and masks.
        save_path: Directory to save prediction plots.
        shape: Height/width of input images.
        in_channels: Number of channels in input images.
        device: Device string ('cpu' or 'cuda').
    """
    images, masks = next(iter(loader))
    ind = np.random.choice(loader.batch_size)

    img_tensor = images[ind].view(1, in_channels, shape, shape).to(device)
    pred = model(img_tensor).cpu().detach()[0].permute(1, 2, 0).reshape(shape, shape)

    image = images[ind].permute(1, 2, 0)[:, :, :min(in_channels, 3)]
    mask = masks[ind].permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(image.numpy())
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask.numpy(), cmap="gray")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(pred.numpy(), cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.tight_layout()
    save_dir = os.path.join(save_path, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{ind}.png"))
    plt.close(fig)
