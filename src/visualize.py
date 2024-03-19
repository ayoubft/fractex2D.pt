import numpy as np
from PIL import Image


def plot_example(loader, ax):
    images, masks = next(iter(loader))
    ind = np.random.choice(range(loader.batch_size))

    image = images[ind].permute(1, 2, 0)[:, :, :3]
    mask = masks[ind]
    mask = Image.fromarray(np.asarray(mask, dtype=np.uint8)).convert('RGB')
    mask = np.asarray(mask)

    ax.imshow(np.hstack([image, mask]))
    ax.set_title('Image / Mask')
    ax.axis('off')
    ax.grid([])


def plot_result(model, loader, ax, shape, in_channels, device):
    images, masks = next(iter(loader))
    ind = np.random.choice(range(loader.batch_size))

    pred = model(
     images[ind].view(1, in_channels, shape, shape).to(device)
    ).cpu().detach()[0].permute(1, 2, 0).reshape(shape, shape)
    # loss = dice2(model(images[ind].view(1, 3, shape, shape).to(device)
    # ), mask[ind].to(device))

    image = images[ind].permute(1, 2, 0)[:, :, :3]
    mask = masks[ind]
    mask = Image.fromarray(
        np.asarray(mask * 255, dtype=np.uint8)).convert('RGB')
    pred = Image.fromarray(
        np.asarray(pred > .5, dtype=np.uint8)).convert('RGB')

    mask = np.asarray(mask)
    pred = np.asarray(pred)

    ax.imshow(np.hstack([image, mask, pred]))
    ax.set_title('Image / Target / Prediction')
    # loss = {round(float(loss), 3)}')
    ax.axis('off')
    ax.grid([])
