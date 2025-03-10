import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io

# TODO: this was on ../

# model_path = '/users/afatihi/work-detect/fractex2D.pt/outputs_2/unet-mse/2025-01-28_12-41/'
# model_path = '/users/afatihi/work-detect/fractex2D.pt/outputs/unet_dice_32/2024-03-10_10-27-58'
model_path = '/users/afatihi/work-detect/fractex2D.pt/outputs_BM/unet-mse/2025-03-07_16-19'
# model_name = 'unet-mse'
model_name = 'test-nicolas'
img_paths = [
    # 'data/ldb/ortho-ldb-z1.png',
    'data/test_ovas/og1-s3',
    # 'data/test_ovas/OG1_sample_3.png',
    # 'data/test_ovas/KL5_sample.png',
    # 'data/test_ovas/KL5_sample_2.png',
    # 'data/test_ovas/wilsons.png',
             ]


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    save_path = model_path

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'),
                          weights_only=True, map_location=torch.device('cpu')))
    # , map_location=torch.device('cpu')  ## torch.load
    model.eval()

    patch_shape = cfg.dataset.shape

    print('init ok')

    for img_path in img_paths:
        img = io.imread(f'{img_path}.png')
        print(img.shape)
        dem = io.imread(f'{img_path}-dem.tif')
        print(dem.shape)
        img = np.concatenate((img[:, :, :3], np.expand_dims(dem, 2)), 2)
        img = img[:512, :512, :]
        print(img.shape)

        print('image ok')

        patch_shape = cfg.dataset.shape
        # SIZE_X = (img.shape[1]//patch_shape)*patch_shape
        # SIZE_Y = (img.shape[0]//patch_shape)*patch_shape
        # img = img[:SIZE_X, :SIZE_Y, :]

        patches = patchify(img, (patch_shape, patch_shape,
                                 cfg.dataset.in_channels), step=256)  # !

        print('patches ok')

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
