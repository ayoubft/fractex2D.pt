import os

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

model_path = '/users/afatihi/work-detect/fractex2D.pt/multirun_BM_/geocrack+/2025-08-04_06-00/dataset.datasets=geocrack-samsu19-matteo21-ovaskainen23'
img_paths = [
    'data/test_ovas/kl5/kl5-s3',
    'data/test_ovas/kl5/hnn-z1-s3',
    'data/test_ovas/kl5/_matteo21-z1-s3',
    'data/test_ovas/kl5/ortho-ldb-z1-s3',
             ]


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    save_path = model_path
    model_name = cfg.model._target_.split('.')[-1]
    print(model_name)
    print(model_path)

    patch_shape = cfg.dataset.shape
    # cfg.threshold = .9

    model = instantiate(cfg.model)
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'),
                          weights_only=True, map_location=torch.device('cpu')))
    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)
    criterion = instantiate(cfg.loss)
    # , map_location=torch.device('cpu')  ## torch.load

    trainloader, valloader, testloader = instantiate(cfg.dataset)

    # ADAPTIVE BATCH NORM
    for epoch in range(2):
        model.train()
        for X_t, _ in tqdm(testloader):
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.track_running_stats = True
            _ = model(X_t.to(device))

    torch.cuda.empty_cache()
    model.eval()
    test_metrics = eval_loop(model, scheduler, criterion, testloader,
                             cfg.threshold, device, model_name)
    print(cfg.dataset.datasets)
    print(test_metrics)

    # all_metrics = []
    # for run in range(1):
    #     test_metrics = eval_loop(model, scheduler, criterion, testloader,
    #                              cfg.threshold, device, model_name,
    #                              cfg.ignore_index)
    #     print(test_metrics)
    #     all_metrics.append(test_metrics)

    # compute mean and std
    # metrics_summary = {}
    # for key in all_metrics[0].keys():
    #     values = [m[key] for m in all_metrics]
    #     metrics_summary[key] = {
    #         "mean": np.mean(values),
    #         "std": np.std(values)
    #     }
    # print(metrics_summary)

    print('init ok')
    # return 0

    for img_path in img_paths:
        img = io.imread(f'{img_path}.png')
        print(img.shape)
        dem = io.imread(f'{img_path}-dem.tif')
        print(dem.shape)
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

        # patchify
        patches = patchify(
            combined_padded, (patch_shape, patch_shape, c), step=patch_shape
        )

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
        pred = np.reshape(
            pred, (patches.shape[0], patches.shape[1], 1, patch_shape, patch_shape, 1)
        )
        pred = unpatchify(pred, combined_padded.shape[:2] + (1,))

        # crop back to original size
        pred = pred[:h, :w, :]

        pred = (pred * 255).astype(np.uint8)
        pred = Image.fromarray(pred.reshape(h, w))

        # patches = patchify(img, (patch_shape, patch_shape,
        #                          cfg.in_channels), step=256)  # !

        # print('patches ok')

        # pred_patches = []
        # for i in range(patches.shape[0]):
        #     for j in range(patches.shape[1]):

        #         single_patch = patches[i, j, :, :, :, :]
        #         single_patch = torch.Tensor(np.array(single_patch))
        #         single_patch = single_patch.permute(0, 3, 1, 2) / 255.

        #         with torch.no_grad():
        #             patch_pred = model(single_patch.to(device))

        #         pred_patches.append(patch_pred.cpu())

        # pred = np.array(pred_patches)
        # pred = np.reshape(pred, (patches.shape[0], patches.shape[1],
        #                          1, patch_shape, patch_shape, 1))  # !
        # pred = unpatchify(pred, (img.shape[0], img.shape[1], 1))

        # pred *= 255
        # pred = Image.fromarray(np.uint8(
        #     pred.reshape(img.shape[0], img.shape[1])))

        # pred = Image.fromarray(np.uint8(pred.reshape(
        #     img.shape[0], img.shape[1])) > cfg.threshold)

        pred.save(os.path.join(save_path,
                  f"{cfg.dataset.datasets}_pred_{img_path.split('/')[-1]}.png"))

    # for img_path in img_paths:
    #     img = Image.open(img_path).convert('RGB')
    #     img_np = np.array(img)

    #     SIZE_X = (img_np.shape[1]//patch_size)*patch_size
    #     SIZE_Y = (img_np.shape[0]//patch_size)*patch_size
    #     img = img.crop((0, 0, SIZE_X, SIZE_Y))

    #     img = np.array(img)

    #     patches = patchify(img, (256, 256, 3), step=256)

    #     pred_patches = []
    #     for i in range(patches.shape[0]):
    #         for j in range(patches.shape[1]):

    #             single_patch = patches[i, j, :, :, :, :]
    #             single_patch = torch.Tensor(np.array(single_patch))
    #             single_patch = single_patch.permute(0, 3, 1, 2)
    #             single_patch /= 255

    #             with torch.no_grad():
    #                 patch_pred = model(single_patch)

    #             pred_patches.append(patch_pred)

    #     pred = np.array(pred_patches)
    #     pred = np.reshape(pred,
    #                      (patches.shape[0], patches.shape[1], 1, 256, 256,1))
    #     pred = unpatchify(pred, (img.shape[0], img.shape[1], 1))

    #     pred *= 255
    #     pred = Image.fromarray(np.uint8(pred.reshape(
    #         img.shape[0], img.shape[1])))

    #     pred.save(os.path.join(model_path, 'pred_' +
    #                            model_name + '_' + img_path.split('/')[-1]))


if __name__ == "__main__":
    main()
