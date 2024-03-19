import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms


def rgbt_dataset(batch_size: int, topo,
                 data_path: str = 'data/jpg',
                 train_root: str = 'train',
                 val_root: str = 'valid',
                 test_root: str = 'test',
                 in_channels=None,
                 out_channels=None,
                 shape=None
                 ):

    trainset = RGBT(dir=data_path, subset=train_root, topo=topo)
    valset = RGBT(dir=data_path, subset=val_root, topo=topo)
    testset = RGBT(dir=data_path, subset=test_root, topo=topo)

    trainloaders = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloaders = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader


class RGBT(Dataset):
    """Load RGB + Topography"""

    def __init__(self, dir: str, subset: str, topo, ext='jpg',
                 transform=None):

        self.topo = topo
        self.images = sorted(
            [os.path.join(dir, subset, 'image', fname)
             for fname in os.listdir(os.path.join(dir, subset, 'image'))
             if fname.endswith(ext)])
        self.masks = sorted(
            [os.path.join(dir, subset, 'gt', fname)
             for fname in os.listdir(os.path.join(dir, subset, 'gt'))
             if fname.endswith(ext)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        mode = 'RGBA' if self.topo else 'RGB'
        image = Image.open(self.images[index]).convert(mode)
        mask = Image.open(self.masks[index]).convert('L')

        image_tensor = torch.from_numpy(np.array(image).astype(np.float32))
        mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32))

        # fix dimensions (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        # scale
        image_tensor /= 255
        mask_tensor /= 255
        mask_tensor[mask_tensor > 0.1] = 1
        mask_tensor[mask_tensor <= 0.1] = 0

        return image_tensor, mask_tensor


# trainloaders, valloaders, testloader = rgbt_dataset(32)
# for (inputs, targets) in (trainloaders):
#     print(inputs, targets.mean())
#     break
