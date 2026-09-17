import json
import os
import warnings

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from patchify import patchify, unpatchify
from PIL import Image
# from ridge_detector import RidgeDetector
from skimage import io
from sklearn.exceptions import UndefinedMetricWarning
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from src.train2 import eval_loop, save_metrics, train_loop
from src.visualize import plot_example, plot_result

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

img_paths = [
    'data/test_ovas/kl5/kl5-s3',
    'data/test_ovas/kl5/hnn-z1-s3',
    'data/test_ovas/kl5/_matteo21-z1-s3',
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
    loss_name = cfg.loss._target_.split('.')[-1]

    writer = SummaryWriter(save_path)

    # 2. Prepare dataset
    trainloader, valloader, testloader = instantiate(cfg.dataset)
    print('Data loaded!')

    if True:
        for _ in range(11):
            plot_example(trainloader, save_path, cfg.in_channels)

    # 3. Define model
    model = instantiate(cfg.model)
    print('Model laoded!')

    # 4. Train model
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    if loss_name == 'BCEWithLogitsLoss':
        criterion = instantiate(
            cfg.loss, pos_weight=torch.Tensor([cfg.param]).to(device))
    else:
        criterion = instantiate(cfg.loss)

    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)

    train_loss = []
    # val_loss = []
    valid_loss_min = np.inf

    for epoch in range(cfg.epochs):
        print(f'Starting epoch {epoch}')
        train_loss = train_loop(
            model, optimizer, criterion, trainloader, device, model_name)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # evaluate on validation set
        model.eval()

        with torch.no_grad():
            running_vloss = 0
            pbar = tqdm(valloader, desc='Iterating over evaluation data')
            for imgs, labels in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                running_vloss += loss.item()*imgs.shape[0]
        running_vloss /= len(valloader.sampler)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': train_loss,
                            'Validation': running_vloss},
                           epoch+1)
        print(f'Epoch {epoch}: loss: {train_loss}, vloss: {running_vloss}')

        # save the model
        if running_vloss <= valid_loss_min:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            valid_loss_min = running_vloss
            print(f"Saving model at this epoch: {epoch} as best model!")

    # save final model
    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))

    # !!. Load best model
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pt'),
                          weights_only=True, map_location=torch.device('cpu')))
    model = model.to(device)

    # 5. Plot test samples
    if True:
        for _ in range(17):
            plot_result(model, testloader, save_path, cfg.dataset.shape,
                        cfg.in_channels, device)

    # ##### Test # #######
    test_metrics = eval_loop(model, scheduler, criterion, testloader,
                             cfg.threshold, device, model_name,
                             cfg.ignore_index)
    save_metrics(test_metrics, 'test', writer, epoch)
    print(test_metrics)

    # 6. Predict test zone
    model.eval()

    if True:
        for img_path in img_paths:
            img = io.imread(f'{img_path}.png')
            dem = io.imread(f'{img_path}-dem.tif')
            # dem = (dem - dem.mean()) / (dem.std() + 1e-8)
            dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
            combined = np.concatenate((img[:, :, :3], np.expand_dims(dem, 2)), 2)

            patch_shape = cfg.dataset.shape
            h, w, c = combined.shape
            pad_h = (patch_shape - h % patch_shape) % patch_shape
            pad_w = (patch_shape - w % patch_shape) % patch_shape

            # pad so divisible by patch size
            combined_padded = np.pad(
                combined,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            patches = patchify(combined_padded, (patch_shape, patch_shape,
                               cfg.in_channels), step=256)  # !

            pred_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):

                    single_patch = patches[i, j, :, :, :, :]
                    single_patch = torch.Tensor(np.array(single_patch))
                    single_patch = single_patch.permute(0, 3, 1, 2) / 255.

                    with torch.no_grad():
                        patch_pred = model(single_patch.to(device))

                    pred_patches.append(patch_pred.cpu())

            pred = np.array(pred_patches)
            pred = np.reshape(pred, (patches.shape[0], patches.shape[1],
                                     1, patch_shape, patch_shape, 1))  # !
            pred = unpatchify(pred, combined_padded.shape[:2] + (1,))

            pred = pred[:h, :w, :]

            pred *= 255
            pred = Image.fromarray(np.uint8(
                pred.reshape(img.shape[0], img.shape[1])))

            pred_proba_path = os.path.join(
                save_path, f"pred_proba_{img_path.split('/')[-1]}.png")
            pred.save(pred_proba_path)


if __name__ == "__main__":
    main()
