import os

import matplotlib.pyplot as plt
import numpy as np


def plot_example(loader, save_path, in_channels):

    fig, axes = plt.subplots(1, max(2, in_channels-1), figsize=(4+in_channels*2, 5))

    images, masks = next(iter(loader))
    ind = np.random.choice(range(loader.batch_size))

    image = images[ind].permute(1, 2, 0)[:, :, :min(in_channels, 3)]
    if in_channels == 4:
        dem = images[ind].permute(1, 2, 0)[:, :, 3]
    mask = masks[ind].permute(1, 2, 0)

    axes[0].imshow(np.array(image))
    axes[0].set_title('Image')
    axes[0].axis('off')
    axes[0].grid([])

    if in_channels == 4:
        axes[1].imshow(dem, cmap='gray')
        axes[1].set_title('DEM')
        axes[1].axis('off')
        axes[1].grid([])

    axes[in_channels-2].imshow(mask, cmap='gray')
    axes[in_channels-2].set_title('Mask')
    axes[in_channels-2].axis('off')
    axes[in_channels-2].grid([])

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'samples/{ind}.png'))
    plt.close()


def plot_result(model, loader, save_path, shape, in_channels, device):

    images, masks = next(iter(loader))
    ind = np.random.choice(range(loader.batch_size))

    pred = model(
     images[ind].view(1, in_channels, shape, shape).to(device)
    ).cpu().detach()[0].permute(1, 2, 0).reshape(shape, shape)
    # loss = dice2(model(images[ind].view(1, 3, shape, shape).to(device)
    # ), mask[ind].to(device))

    image = images[ind].permute(1, 2, 0)[:, :, :min(in_channels, 3)]
    mask = masks[ind].permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].imshow(np.array(image))
    axes[0].set_title('Image')
    axes[0].axis('off')
    axes[0].grid([])

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground truth')
    axes[1].axis('off')
    axes[1].grid([])

    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    axes[2].grid([])

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'predictions/{ind}.png'))
    plt.close()
