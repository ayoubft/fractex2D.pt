import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from patchify import patchify, unpatchify
from PIL import Image


model_path = 'outputs_tmp/unet_rgb_dice_ova23_32/2024-04-02_14-25/'
img_path = 'data/test_ovas/OG1_sample.png'


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    model.eval()

    img = Image.open(img_path)
    img = np.array(img)

    patches = patchify(img, (256, 256, 3), step=256)

    pred_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):

            single_patch = patches[i, j, :, :, :, :]
            single_patch = torch.Tensor(np.array(single_patch))
            single_patch = single_patch.permute(0, 3, 1, 2)
            single_patch /= 255

            with torch.no_grad():
                patch_pred = model(single_patch)

            pred_patches.append(patch_pred)

    pred = np.array(pred_patches)
    pred = np.reshape(pred,
                      (patches.shape[0], patches.shape[1], 1, 256, 256, 1))
    pred = unpatchify(pred, (img.shape[0], img.shape[1], 1))

    pred *= 255
    pred = Image.fromarray(np.uint8(pred.reshape(img.shape[0], img.shape[1])))

    pred.save(os.path.join(model_path, 'pred_' + img_path.split('/')[-1]))


if __name__ == "__main__":
    main()
