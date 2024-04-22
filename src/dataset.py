import os

import numpy as np
import torch
import torchvision.transforms as t
from PIL import Image
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms


def rgbt_dataset(batch_size: int,
                 topo,
                 list,
                 ext,
                 data_path: str = 'data/jpg',
                 train_root: str = 'train',
                 val_root: str = 'valid',
                 test_root: str = 'test',
                 aug_mult: int = 1,
                 in_channels=None,
                 out_channels=None,
                 shape=None
                 ):

    transforms = None
    _ = t.Compose([
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        # t.RandomRotation(15)
    ])

    trainset = RGBT(dir=data_path, subset=train_root, topo=topo, list=list,
                    ext=ext, transform=transforms, aug_mult=aug_mult)
    valset = RGBT(dir=data_path, subset=val_root, topo=topo, list=list,
                  ext=ext, transform=transforms, aug_mult=aug_mult)
    testset = RGBT(dir=data_path, subset=test_root, topo=topo, list=list,
                   ext=ext, transform=transforms, aug_mult=aug_mult)

    trainloaders = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloaders = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader


class RGBT(Dataset):
    """Load RGB + Topography"""

    def __init__(self, dir: str, subset: str, topo, list=False, ext='jpg',
                 transform=None, aug_mult=1):

        if list:
            fnames = []
            with open(os.path.join(dir, subset, 'list.txt'), 'r') as f:
                for line in f:
                    fnames.append(line.strip())

            self.images = sorted(
                [os.path.join(dir, subset, 'image', fname)
                 for fname in fnames if fname.endswith(ext)])
            self.masks = sorted(
                [os.path.join(dir, subset, 'gt', fname)
                 for fname in fnames if fname.endswith(ext)])

        if not list:
            self.images = sorted(
                [os.path.join(dir, subset, 'image', fname)
                 for fname in os.listdir(os.path.join(dir, subset, 'image'))
                 if fname.endswith(ext)])
            self.masks = sorted(
                [os.path.join(dir, subset, 'gt', fname)
                 for fname in os.listdir(os.path.join(dir, subset, 'gt'))
                 if fname.endswith(ext)])

        self.topo = topo
        self.transform = transform
        self.aug_mult = aug_mult

    def __len__(self):
        return len(self.images) * self.aug_mult

    def __getitem__(self, index):

        data_idx = index % len(self.images)

        mode = 'RGBA' if self.topo else 'RGB'
        image = Image.open(self.images[data_idx]).convert(mode)
        mask = Image.open(self.masks[data_idx]).convert('L')

        image_tensor = torch.from_numpy(np.array(image).astype(np.float32))
        mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32))

        # fix dimensions (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        # scale
        image_tensor /= 255
        mask_tensor /= 255
        mask_tensor[mask_tensor > 0.01] = 1
        mask_tensor[mask_tensor <= 0.01] = 0

        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor, mask_tensor
