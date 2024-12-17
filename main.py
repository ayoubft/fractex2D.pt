import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from patchify import patchify, unpatchify
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from src.train import eval_loop, train_loop
from src.visualize import plot_example, plot_result

img_paths = [
    'data/test_ovas/OG1_sample_3.png',
    'data/test_ovas/KL5_sample.png',
    'data/test_ovas/KL5_sample_2.png',
    'data/test_ovas/wilsons.png',
             ]


@hydra.main(config_name="config.yaml", config_path="config", version_base=None)
def main(cfg: DictConfig):

    # 1. Parse config & get experiment output dir
    print('Config:\n', OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    # os.mkdir(os.path.join(save_path, 'test'))
    print('Output dir:\n', save_path)

    writer = SummaryWriter(save_path)

    # 2. Prepare dataset
    trainloader, valloader, testloader = instantiate(cfg.dataset)

    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for ax in axes.flatten():
        plot_example(trainloader, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'samples.png'))

    # 3. Define model
    model = instantiate(cfg.model)

    # 4. Train model
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    criterion = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, model.parameters())

    train_loss = []
    # val_loss = []
    valid_loss_min = np.inf

    for epoch in range(cfg.epochs):
        train_loss = train_loop(
            model, optimizer, criterion, trainloader, device)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # evaluate on validation set
        metrics = eval_loop(model, criterion, valloader, cfg.threshold, device)
        writer.add_scalar("Loss/valid", metrics['loss'], epoch)
        writer.add_scalar("ACC/valid", metrics['accuracy'], epoch)
        writer.add_scalar("F1/valid", metrics['f1_macro'], epoch)

        # show progress
        print_string = f'Epoch: {epoch+1}, TrainLoss: {train_loss:.5f}, \
            ValidLoss: {metrics["loss"]:.5f}, ACC: {metrics["accuracy"]:.5f}, \
            F1: {metrics["f1_macro"]:.3f}'
        print(print_string)

        # save the model
        if metrics["loss"] <= valid_loss_min:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            # be sure to call model.eval() method before inferencing
            valid_loss_min = metrics["loss"]

    # 5. Plot test samples
    fig, axes = plt.subplots(3, 3, figsize=(13, 5))
    for ax in axes.flatten():
        plot_result(model, testloader, ax, cfg.dataset.shape,
                    cfg.dataset.in_channels, device)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'test.png'))
    writer.flush()

    # 6. Predict test zone
    model.eval()

    for img_path in img_paths:
        img = Image.open(img_path)
        img = np.array(img)

        patch_shape = cfg.dataset.shape
        patches = patchify(img, (patch_shape, patch_shape,
                                 cfg.dataset.in_channels), step=256)  # !

        pred_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):

                single_patch = patches[i, j, :, :, :, :]
                single_patch = torch.Tensor(np.array(single_patch))
                single_patch = single_patch.permute(0, 3, 1, 2)
                single_patch /= 255

                with torch.no_grad():
                    patch_pred = model(single_patch.to(device))

                pred_patches.append(patch_pred.cpu())

        pred = np.array(pred_patches)
        pred = np.reshape(pred, (patches.shape[0], patches.shape[1],
                                 1, patch_shape, patch_shape, 1))  # !
        pred = unpatchify(pred, (img.shape[0], img.shape[1], 1))

        pred *= 255
        pred = Image.fromarray(np.uint8(pred.reshape(
            img.shape[0], img.shape[1])) > cfg.threshold)

        pred.save(os.path.join(save_path, 'pred_' +
                               img_path.split('/')[-1]))


if __name__ == "__main__":
    main()
