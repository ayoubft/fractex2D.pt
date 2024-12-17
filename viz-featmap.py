import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

model_path = '/users/afatihi/work-detect/fractex2D.pt/outputs_tmp/unet_rgb_dice_ova23_32/2024-04-02_14-25'
model_name = 'unet-dice'

@hydra.main(config_name="config.yaml", config_path="config", version_base=None)
def main(cfg: DictConfig):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    trainloader, valloader, testloader = instantiate(cfg.dataset)
    data = next(iter(trainloader)) # one batch

    # load model
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    model.to(device)
    model.eval()

    trainloader, valloader, testloader = instantiate(cfg.dataset)

    print(model.encoder1.enc1conv1)

    def normalize_output(img):
        img = img - img.min()
        img = img / img.max()
        return img


    data = data[0][0]
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)
    print(output.size, output.shape)
    
    # Plot some images
    idx = torch.randint(0, output.size(0), ())
    pred = normalize_output(output[idx, 0])
    img = data[idx, 0]

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img.detach().cpu().numpy())
    axarr[1].imshow(pred.detach().cpu().numpy())

    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.encoder1.enc1conv1.register_forward_hook(get_activation('enc1conv1'))

    data = data[0][0]
    data.unsqueeze_(0)
    data = data.to(device)
    output = model(data)
    
    act = activation['enc1conv1'].squeeze()
    num_filters = act.size(0)  # Number of filters (channels)
    grid_cols = math.ceil(math.sqrt(num_filters))  # Number of columns in the grid
    grid_rows = math.ceil(num_filters / grid_cols)  # Number of rows in the grid
    
    # Create subplots
    fig, axarr = plt.subplots(grid_rows, grid_cols, figsize=(16, 16))
    
    # Flatten axes for easy iteration
    axarr = axarr.flatten()

    for idx in range(num_filters):
        axarr[idx].imshow(act[idx].cpu())

    print('saving')
    plt.savefig('viz-featmap.png')


if __name__ == "__main__":
    main()
