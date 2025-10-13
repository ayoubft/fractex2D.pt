import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


model_path = '/users/afatihi/work-detect/fractex2D.pt/outputs_BM_/unet-huber-rmspro-0.1/2025-08-07_09-28'


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
                      f'{model_path}/model.onnx',  # where to save the model
                      export_params=True,
                      opset_version=15,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


if __name__ == "__main__":
    main()
