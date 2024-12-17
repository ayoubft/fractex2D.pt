import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.smooth_tiled_predictions import predict_img_with_smooth_windowing
from PIL import Image

# TODO: this was on ../

model_path = 'outputs_tmp/unet_rgb_dice_ova23_32/2024-04-02_14-25'
model_name = 'unet'
img_paths = [
    # 'data/test_ovas/OG1_sample_3.png',
    # 'data/test_ovas/KL5_sample.png',
    'data/test_ovas/KL5_sample_2.png',
    # 'data/test_ovas/wilsons.png',
             ]


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    model.eval()

    patch_size = cfg.dataset.shape

    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        predictions_smooth = predict_img_with_smooth_windowing(
            img,
            window_size=patch_size,
            subdivisions=2,  # Minimal amount of overlap for windowing §even§
            nb_classes=1,
            pred_func=(
                lambda img_batch_subdiv: model(img_batch_subdiv)
            )
        )

        final_prediction = np.argmax(predictions_smooth, axis=2)
        print(final_prediction)

        # pred.save(os.path.join(model_path, 'pred_' +
        #                        model_name + '_' + img_path.split('/')[-1]))


if __name__ == "__main__":
    main()
