import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


model_path = '/users/afatihi/work-detect/fractex2D.pt/outputs_tmp/unet_rgb_dice_ova23_32/2024-04-02_14-25'
model_name = 'unet-dice'
img_paths = [
    'data/ldb/ortho-ldb-z1.png',
    # 'data/test_ovas/OG1_sample_3.png',
    # 'data/test_ovas/KL5_sample.png',
    # 'data/test_ovas/KL5_sample_2.png',
    # 'data/test_ovas/wilsons.png',
             ]


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    model = instantiate(cfg.model)
    # model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    # model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')['state_dict'])
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'),
                                     map_location='cpu'))
    model.eval()

    patch_size = cfg.dataset.shape
    in_channels = cfg.dataset.in_channels

    x = torch.rand(1, in_channels, patch_size, patch_size)
    _ = model(x)

    torch.onnx.export(model,
                      x,  # model input
                      f'onnx/{model_name}.onnx',  # where to save the model
                      export_params=True,
                      opset_version=15,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


if __name__ == "__main__":
    main()
