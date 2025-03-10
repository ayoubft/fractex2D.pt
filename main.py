import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from src.train import eval_loop, train_loop
from src.visualize import plot_example, plot_result

img_paths = [
    'data/test_ovas/og1-s3',
             ]


@hydra.main(config_name="config.yaml", config_path="config", version_base=None)
def main(cfg: DictConfig):

    # 1. Parse config & get experiment output dir
    print('Config:\n', OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    os.mkdir(os.path.join(save_path, 'samples'))
    os.mkdir(os.path.join(save_path, 'predictions'))
    print('Output dir:\n', save_path)
    model_name = cfg.model._target_.split('.')[-1]

    writer = SummaryWriter(save_path)

    # 2. Prepare dataset
    trainloader, valloader, testloader = instantiate(cfg.dataset)

    if True:
        for _ in range(11):
            plot_example(trainloader, save_path)

    # 3. Define model
    model = instantiate(cfg.model)

    # 4. Train model
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    criterion = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)

    train_loss = []
    # val_loss = []
    valid_loss_min = np.inf

    for epoch in range(cfg.epochs):
        train_loss = train_loop(
            model, optimizer, criterion, trainloader, device, model_name)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # evaluate on validation set
        before_lr = optimizer.param_groups[0]["lr"]
        metrics = eval_loop(model, scheduler, criterion, valloader,
                            cfg.threshold, device, model_name)
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: LR %.4f -> %.4f" % (epoch, before_lr, after_lr))

        writer.add_scalar("Loss/valid", metrics['loss'], epoch)
        writer.add_scalar("MSE/valid", metrics['mse'], epoch)
        writer.add_scalar("PSNR/valid", metrics['psnr'], epoch)
        writer.add_scalar("SSIM/valid", metrics['ssim'], epoch)
        writer.add_scalar("AE/valid", metrics['ae'], epoch)

        # show progress
        print_string = f'Epoch: {epoch+1}, TrainLoss: {train_loss:.5f}, \
            ValidLoss: {metrics["loss"]:.5f}, MSE: {metrics["mse"]:.5f}, \
            PSNR: {metrics["psnr"]:.3f}, SSIM: {metrics["ssim"]:.5f},\
            AE: {metrics["ae"]:.5f}'
        print(print_string)

        # save the model
        if metrics["loss"] <= valid_loss_min:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            # be sure to call model.eval() method before inferencing
            valid_loss_min = metrics["loss"]

    # 5. Plot test samples
    if True:
        for _ in range(11):
            plot_result(model, testloader, save_path, cfg.dataset.shape,
                        cfg.dataset.in_channels, device)

    # 6. Predict test zone
    model.eval()

    if True:
        for img_path in img_paths:
            img = io.imread(f'{img_path}.png')
            dem = io.imread(f'{img_path}-dem.tif')
            img = np.concatenate((img[:, :, :3], np.expand_dims(dem, 2)), 2)

            patch_shape = cfg.dataset.shape
            # SIZE_X = (img.shape[1]//patch_shape)*patch_shape
            # SIZE_Y = (img.shape[0]//patch_shape)*patch_shape
            # img = img[:SIZE_X, :SIZE_Y, :]

            patches = patchify(img, (patch_shape, patch_shape,
                                     cfg.dataset.in_channels), step=256)  # !

            pred_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):

                    single_patch = patches[i, j, :, :, :, :]
                    single_patch = torch.Tensor(np.array(single_patch))
                    single_patch = single_patch.permute(0, 3, 1, 2)
                    # single_patch /= 255

                    with torch.no_grad():
                        patch_pred = model(single_patch.to(device))

                    pred_patches.append(patch_pred.cpu())

            pred = np.array(pred_patches)
            pred = np.reshape(pred, (patches.shape[0], patches.shape[1],
                                     1, patch_shape, patch_shape, 1))  # !
            pred = unpatchify(pred, (img.shape[0], img.shape[1], 1))

            pred *= 255
            pred = Image.fromarray(np.uint8(
                pred.reshape(img.shape[0], img.shape[1])))

            # pred = Image.fromarray(np.uint8(pred.reshape(
            #     img.shape[0], img.shape[1])) > cfg.threshold)
            pred.save(os.path.join(save_path,
                      f"pred_{img_path.split('/')[-1]}.png"))
            # break


if __name__ == "__main__":
    main()
