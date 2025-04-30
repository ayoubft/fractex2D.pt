import os

import numpy as np
import torch
import torchvision.transforms as t
from PIL import Image
from skimage import io
from skimage.filters.rank import maximum
from skimage.measure import label
from skimage.morphology import binary_dilation, dilation, disk, square
from skimage.segmentation import expand_labels
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def expand_wide_fractures_gt(img, gt, disk_size=2, thresh=30, gt_thresh=100, gt_ext='png'):
    """
    Expand the ground truth (gt) mask to include wide fractures in the image.
    This function uses a maximum filter to identify large areas of dark pixels in the image,
    and then dilates the gt mask to include these areas. The function also checks for spatial
    contiguity with the gt mask, and only keeps objects that are spatially contiguous with
    objects in the gt mask.

    Parameters
    -----------
    img (numpy.ndarray): The input image.
    gt (numpy.ndarray): The ground truth mask.
    disk_size (int): The size of the disk used for morphological operations.
    thresh (int): The threshold value for the maximum filter.
    gt_thresh (int): The threshold value for the ground truth mask.
    gt_ext (str): The file extension for the ground truth mask.

    Returns
    -----------
    numpy.ndarray: The expanded ground truth mask.
    Thanks to Sam Thiele
    """
    gray = img[...,1] # convert to greyscale (use green channel so vegetation is not so dark)
    imax = maximum( gray, disk(disk_size)) # use a maximum filter to only keep large areas of dark pixels
    #thresh = np.min(imax) * tmul # threshold based on the minimum remaining value in the image
    msk = binary_dilation( imax < thresh, disk(disk_size)) # dilate again after threshold to "re-fill" the dark areas
    
    # add in gt again
    gtg = np.logical_or( msk, gt > gt_thresh )
    
    # only keep objects that are spatially contiguous with objects in the gt
    labeled_components, num_components = label(gtg, connectivity=1, return_num=True)
    overlap_check = []
    for component_id in range(1, num_components + 1): # Check overlap with gt
        component_mask = labeled_components == component_id
        if not np.any(gt[component_mask] > gt_thresh):
            gtg[component_mask] = 0

    if 'tif' in gt_ext:
        new_gt = np.array(gt*255, dtype=np.uint8) | np.array(gtg*255, dtype=np.uint8)
    else:
        new_gt = np.array(gt, dtype=np.uint8) | np.array(gtg*255, dtype=np.uint8)
    return new_gt


def dilate_labels(image):
    """
    Apply multi-scale dilation to labeled regions in an image.

    This function expands labeled regions in a segmented image and applies
    three levels of morphological dilation using structuring elements of increasing sizes.
    Each dilation mask is added to the original expanded image with decreasing weights.

    Parameters
    ------------
    image (np.ndarray): The input image

    Returns
    ------------
    np.ndarray: The image with smoothed label boundaries.
    """
    # Expand labeled regions to fill small gaps between labels
    expanded = expand_labels(image, distance=2)

    # Perform three levels of dilation and create masks for each scale
    dilated_mask_1 = dilation(expanded, square(5)) ^ expanded
    dilated_mask_2 = dilation(expanded, square(9)) ^ dilated_mask_1 ^ expanded
    dilated_mask_3 = dilation(expanded, square(12)) ^ dilated_mask_2 ^ dilated_mask_1 ^ expanded

    # Blend the original and dilated masks with decreasing influence
    blended = expanded + dilated_mask_1 / 3 + dilated_mask_2 / 5 + dilated_mask_3 / 9

    # Convert to unsigned 8-bit integer format for compatibility
    output = np.array(blended, dtype=np.uint8)

    return output


class RGBT(Dataset):
    """Load RGB + Topography"""

    def __init__(self, directory: str, subset: str, topo, use_list=False, ext='jpg',
                 transform=None):

        if use_list:
            fnames = []
            with open(os.path.join(directory, subset, 'list.txt'), 'r') as f:
                for line in f:
                    fnames.append(line.strip())

            self.images = sorted(
                [os.path.join(directory, subset, 'image', fname)
                 for fname in fnames if fname.endswith(ext)])
            self.masks = sorted(
                [os.path.join(directory, subset, 'gt', fname)
                 for fname in fnames if fname.endswith(ext)])

        if not use_list:
            self.images = sorted(
                [os.path.join(directory, subset, 'image', fname)
                 for fname in os.use_listdirectory(os.path.join(directory, subset, 'image'))
                 if fname.endswith(ext)])
            self.masks = sorted(
                [os.path.join(directory, subset, 'gt', fname)
                 for fname in os.use_listdirectory(os.path.join(directory, subset, 'gt'))
                 if fname.endswith(ext)])

        self.topo = topo
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        data_idx = index % len(self.images)

        mode = 'RGBA' if self.topo else 'RGB'
        image = Image.open(self.images[data_idx]).convert(mode)
        mask = Image.open(self.masks[data_idx]).convert('L')

        image_tensor = torch.from_numpy(np.array(image).astype(np.float32))
        mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32))
        mask_tensor.unsqueeze_(0)

        # fix dimensions (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        # scale
        image_tensor /= 255
        mask_tensor /= 255
        # mask_tensor[mask_tensor > 0.01] = 1
        # mask_tensor[mask_tensor <= 0.01] = 0

        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor, mask_tensor


class OVAS(Dataset):

    def __init__(self, subset: str, use_list=True, topo=False, transform=None):

        directory = 'data/ovaskainen23_'
        ext_img = 'png'
        ext_ = 'tif'

        if use_list:
            fnames = []
            with open(os.path.join(directory, subset, 'list.txt'), 'r') as f:
                for line in f:
                    fnames.append(line.strip())

            self.images = sorted(
                [file.replace(ext_, ext_img)
                    for file in [os.path.join(directory, subset, 'image', fname)
                                 for fname in fnames if fname.endswith(ext_)]
                 ])
            self.masks = sorted(
                [os.path.join(directory, subset, 'gt', fname)
                 for fname in fnames if fname.endswith(ext_)])
            self.dems = sorted(
                [os.path.join(directory, subset, 'dem', fname)
                 for fname in fnames if fname.endswith(ext_)])

        if not use_list:  #! update for DEM
            self.images = sorted(
                [os.path.join(directory, subset, 'image', fname)
                 for fname in os.listdir(os.path.join(directory, subset, 'image'))
                 if fname.endswith(ext)])
            self.masks = sorted(
                [os.path.join(directory, subset, 'gt', fname)
                 for fname in os.listdir(os.path.join(directory, subset, 'gt'))
                 if fname.endswith(ext)])

        self.topo = topo
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        data_idx = index % len(self.images)

        # mode = 'RGBA' if self.topo else 'RGB'
        image = io.imread(self.images[data_idx])[:, :, :3]
        gt = (io.imread(self.masks[data_idx])*255).astype(np.uint8)
        gt = expand_wide_fractures_gt(image[:, :, :3].astype(np.uint8), gt)
        gt = dilate_labels(gt)
        dem = io.imread(self.dems[data_idx])
        # image = Image.open(self.images[data_idx]).convert(mode)
        # mask = Image.open(self.masks[data_idx]).convert('L')

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(gt).unsqueeze(0)
        dem_tensor = torch.from_numpy(dem)

        image_tensor = torch.cat((image_tensor, dem_tensor.unsqueeze(2)), 2)

        # fix dimensions (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        # scale
        # image_tensor /= 255
        # mask_tensor /= 255
        # mask_tensor[mask_tensor > 0.01] = 1
        # mask_tensor[mask_tensor <= 0.01] = 0

        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor.float(), mask_tensor.float()


class MATTEO(Dataset):

    def __init__(self, subset: str, use_list=True, topo=False, transform=None):

        directory = 'data/matteo21'
        ext = 'tif'

        self.images = sorted(
            [os.path.join(directory, subset, 'image', fname)
                for fname in os.listdir(os.path.join(directory, subset, 'image'))
                if fname.endswith(ext)])
        self.masks = sorted(
            [os.path.join(directory, subset, 'gt', fname)
                for fname in os.listdir(os.path.join(directory, subset, 'gt'))
                if fname.endswith(ext)])

        self.topo = topo
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        data_idx = index % len(self.images)

        image = io.imread(self.images[data_idx])
        gt = (io.imread(self.masks[data_idx])*255).astype(np.uint8)
        gt = expand_wide_fractures_gt(image[:, :, :3].astype(np.uint8), gt)
        gt = dilate_labels(gt)

        image_tensor = torch.from_numpy(image)  #[:, :, :3]
        # dem_tensor = torch.from_numpy(image)[:, :, 3].unsqueeze(0)
        mask_tensor = torch.from_numpy(gt).unsqueeze(0)

        # image_tensor = torch.cat((image_tensor, dem_tensor.unsqueeze(2)), 2)

        # fix dimensions (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor.float(), mask_tensor.float()


class SAMSU(Dataset):

    def __init__(self, subset: str, use_list=True, topo=False, transform=None):

        directory = 'data/samsu19'
        ext_img = 'png'
        ext_ = 'tif'

        if use_list:
            fnames = []
            with open(os.path.join(directory, subset, 'list.txt'), 'r') as f:
                for line in f:
                    fnames.append(line.strip())

            self.images = sorted(
                [file.replace(ext_, ext_img)
                    for file in [os.path.join(directory, subset, 'image', fname)
                                 for fname in fnames if fname.endswith(ext_)]
                 ])
            self.masks = sorted(
                [os.path.join(directory, subset, 'gt', fname)
                 for fname in fnames if fname.endswith(ext_)])
            self.dems = sorted(
                [os.path.join(directory, subset, 'dem', fname)
                 for fname in fnames if fname.endswith(ext_)])

        if not use_list:  #! update for DEM
            self.images = sorted(
                [os.path.join(directory, subset, 'image', fname)
                 for fname in os.listdir(os.path.join(directory, subset, 'image'))
                 if fname.endswith(ext)])
            self.masks = sorted(
                [os.path.join(directory, subset, 'gt', fname)
                 for fname in os.listdir(os.path.join(directory, subset, 'gt'))
                 if fname.endswith(ext)])

        self.topo = topo
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        data_idx = index % len(self.images)

        # mode = 'RGBA' if self.topo else 'RGB'
        image = io.imread(self.images[data_idx])[:, :, :3]
        gt = (io.imread(self.masks[data_idx])*255).astype(np.uint8)
        # gt = expand_wide_fractures_gt(image[:, :, :3].astype(np.uint8), gt)
        gt = dilate_labels(gt)
        dem = io.imread(self.dems[data_idx])
        # image = Image.open(self.images[data_idx]).convert(mode)
        # mask = Image.open(self.masks[data_idx]).convert('L')

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(gt).unsqueeze(0)
        dem_tensor = torch.from_numpy(dem)

        image_tensor = torch.cat((image_tensor, dem_tensor.unsqueeze(2)), 2)

        # fix dimensions (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)

        # scale
        # image_tensor /= 255
        # mask_tensor /= 255
        # mask_tensor[mask_tensor > 0.01] = 1
        # mask_tensor[mask_tensor <= 0.01] = 0

        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor.float(), mask_tensor.float()


DATASETS = {
    'ovaskainen23': OVAS,
    'matteo21': MATTEO,
    'samsu19': SAMSU,
}


def all_datasets(batch_size: int = 32,
                 datasets: list = ['samsu19', 'matteo21', 'ovaskainen23'],
                 in_channels: int = 4,  # #! to change
                 out_channels: int = 1,
                 shape: int = 256,
                 ):

    transforms = t.Compose([
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        t.ColorJitter(brightness=.5, hue=.3),
        t.RandomAutocontrast(),
        t.RandomAdjustSharpness(sharpness_factor=2),
        t.RandomRotation(degrees=(15, 70)),
    ])

    all_train = []
    all_val = []
    all_test = []

    for name in datasets:
        trainset_ = DATASETS[name]('train', transforms)
        all_train.append(trainset_)

        valset_ = DATASETS[name]('valid')
        all_val.append(valset_)

        testset_ = DATASETS[name]('test')
        all_test.append(testset_)

    trainset = ConcatDataset(all_train)
    valset = ConcatDataset(all_val)
    testset = ConcatDataset(all_test)

    trainloaders = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloaders = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader


def rgbt_dataset(batch_size: int,
                 topo,
                 use_list,
                 ext,
                 data_path: list = ['data/jpg', 'data/jpg'],
                 train_root: str = 'train',
                 val_root: str = 'valid',
                 test_root: str = 'test',
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

    trainset = RGBT(directory=data_path, subset=train_root, topo=topo,
                    use_list=use_list, ext=ext, transform=transforms)
    valset = RGBT(directory=data_path, subset=val_root, topo=topo, use_list=use_list,
                  ext=ext, transform=transforms)
    testset = RGBT(directory=data_path, subset=test_root, topo=topo,
                   use_list=use_list, ext=ext, transform=transforms)

    trainloaders = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloaders = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader
