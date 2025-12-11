from pathlib import Path
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as t
import torchvision.transforms.v2.functional as TF
from skimage import io
from skimage.filters.rank import maximum
from skimage.measure import label
from skimage.morphology import binary_dilation, dilation, disk
from skimage.segmentation import expand_labels
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# -------------------------
# Label pre-processing
# -------------------------
def expand_wide_fractures_gt(
    img: np.ndarray,
    gt: np.ndarray,
    disk_size: int = 2,
    thresh: int = 30,
    gt_thresh: int = 100,
    gt_ext: str = "png",
) -> np.ndarray:
    """
    Expand a binary/soft ground-truth mask to include nearby wide/dark fractures.

    Method:
      - Use green channel (index 1) as a grayscale proxy.
      - Apply a maximum filter to emphasize large dark regions.
      - Threshold and dilate to form a candidate mask.
      - Keep only connected components that overlap the original GT.
      - Return a combined mask as uint8 (0..255). If gt_ext contains "tif" the
        original `gt` is assumed to be already in [0,1] or in the original dtype;
        the code preserves existing scaling behavior from the original script.

    Args:
        img: HxWxC image (expects at least 2 channels; green channel used).
        gt: HxW ground-truth mask (expected in [0..1] or [0..255]).
        disk_size: radius for morphological operations.
        thresh: threshold applied to the maximum-filtered gray image.
        gt_thresh: threshold to consider a pixel part of the original GT.
        gt_ext: file extension of GT (affects final combination step).

    Returns:
        Expanded GT mask as np.uint8 (values 0 or 255).
    """
    if img.ndim < 3 or img.shape[2] < 2:
        raise ValueError("img must have at least 2 channels (uses green channel).")

    # use green channel as grayscale proxy
    gray = img[..., 1].astype(np.uint8)

    # keep large dark areas via maximum filter, then threshold and dilate
    imax = maximum(gray, disk(disk_size))
    candidate = binary_dilation(imax < thresh, disk(disk_size))

    # combine candidate with existing GT (considering gt_thresh)
    gt_bool = gt > gt_thresh
    combined = np.logical_or(candidate, gt_bool)

    # remove connected components that do not overlap original GT
    labeled, num = label(combined, connectivity=1, return_num=True)
    for comp_id in range(1, num + 1):
        comp_mask = labeled == comp_id
        if not np.any(gt_bool[comp_mask]):
            combined[comp_mask] = False

    # produce uint8 [0,255] result with behavior matching original code
    if "tif" in gt_ext:
        # preserve original gt scaling behavior from source
        new_gt = (np.array(gt * 255, dtype=np.uint8) | np.array(combined * 255, dtype=np.uint8))
    else:
        new_gt = (np.array(gt, dtype=np.uint8) | np.array(combined * 255, dtype=np.uint8))

    return new_gt


def dilate_labels(image: np.ndarray) -> np.ndarray:
    """
    Smooth label boundaries by multi-scale dilation and blending.

    - Expand labels to fill tiny gaps (expand_labels).
    - Create three dilation masks with increasing disks and blend them into
      a smoothed label map with decreasing weights.

    Args:
        image: integer-labeled image or binary mask (HxW).

    Returns:
        np.uint8 array (HxW) with blended/smoothed label boundaries.
    """
    expanded = expand_labels(image, distance=2)

    # Multi-scale dilation masks (exclusive differences)
    d1 = dilation(expanded, disk(2)) ^ expanded
    d2 = dilation(expanded, disk(5)) ^ d1 ^ expanded
    d3 = dilation(expanded, disk(7)) ^ d2 ^ d1 ^ expanded

    blended = expanded + d1 / 3.0 + d2 / 5.0 + d3 / 9.0
    return np.array(blended, dtype=np.uint8)


# -------------------------
# Augmentation helpers
# -------------------------
def _apply_random_flips(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal and vertical flips (50% each)."""
    if random.random() > 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if random.random() > 0.5:
        image, mask = TF.vflip(image), TF.vflip(mask)
    return image, mask


def _apply_random_photometric_augmentations(image: torch.Tensor, prob_config: Optional[dict] = None) -> torch.Tensor:
    """
    Photometric augmentations applied independently with small probabilities.

    The function preserves an extra channel (e.g. DEM) if image has 4 channels:
      - augment only the first three (RGB) channels, then concatenate the extra.
    """
    if prob_config is None:
        prob_config = {
            "gaussian_blur": 0.05,
            "darken_low": 0.05,
            "brighten": 0.15,
            "contrast": 0.05,
            "saturation": 0.05,
        }

    has_extra = image.shape[0] == 4
    rgb = image[:3] if has_extra else image

    # gaussian blur
    if random.random() < prob_config["gaussian_blur"]:
        sigma = random.uniform(0.1, 2.0)
        rgb = TF.gaussian_blur(rgb, kernel_size=5, sigma=sigma)

    # darken (factor < 1)
    if random.random() < prob_config["darken_low"]:
        factor = random.uniform(0.7, 0.9)
        rgb = TF.adjust_brightness(rgb, factor)

    # brighten (factor > 1)
    if random.random() < prob_config["brighten"]:
        factor = random.uniform(1.1, 1.7)
        rgb = TF.adjust_brightness(rgb, factor)

    # contrast
    if random.random() < prob_config["contrast"]:
        factor = random.uniform(0.7, 1.5)
        rgb = TF.adjust_contrast(rgb, factor)

    # saturation
    if random.random() < prob_config["saturation"]:
        factor = random.uniform(0.7, 1.5)
        rgb = TF.adjust_saturation(rgb, factor)

    if has_extra:
        image = torch.cat([rgb, image[3:]], dim=0)
    else:
        image = rgb

    return image


# -------------------------
# Base dataset utilities
# -------------------------
def _read_image(path: Path) -> np.ndarray:
    """Read image with skimage.io and ensure dtype uint8."""
    arr = io.imread(str(path))
    # convert floats to uint8 if necessary
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _read_mask(path: Path) -> np.ndarray:
    """Read mask and convert to uint8 0..255."""
    arr = io.imread(str(path))
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


# -------------------------
# Dataset classes
# -------------------------
class BaseCrackDataset(Dataset):
    """
    Minimal common functionality for the specific dataset wrappers used downstream.

    Subclasses must provide:
      - self.images (list[Path])
      - self.masks  (list[Path])
      - optional self.dems  (list[Path]) when in_channels==4
    """

    def __init__(
        self,
        images: Sequence[Path],
        masks: Sequence[Path],
        dem_paths: Optional[Sequence[Path]] = None,
        topo: bool = False,
        transform: bool = False,
        expand: bool = True,
        dilate: bool = True,
        in_channels: int = 3,
    ):
        self.images = list(images)
        self.masks = list(masks)
        self.dems = list(dem_paths) if dem_paths is not None else None

        self.topo = topo
        self.transform = transform
        self.expand = expand
        self.dilate = dilate
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.images)

    def _load_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image/mask pair, apply optional expand/dilate and channel handling,
        then perform flips and photometric augmentations.
        """
        img_np = _read_image(Path(self.images[idx]))
        gt_np = _read_mask(Path(self.masks[idx]))

        # expand wide fractures (if requested)
        if self.expand:
            gt_np = expand_wide_fractures_gt(img_np[:, :, :3].astype(np.uint8), gt_np)

        # dilate labels (if requested)
        if self.dilate:
            gt_np = dilate_labels(gt_np)

        # build image tensor. If dataset provides DEM as a separate file, append as 4th channel.
        img_tensor = torch.from_numpy(img_np[:, :, :3])
        if self.in_channels == 4:
            # if DEM present inside the image array or as separate file, handle both cases
            if img_np.shape[2] >= 4:
                dem_np = img_np[:, :, 3].astype(np.float32)
            elif self.dems is not None:
                dem_np = _read_image(Path(self.dems[idx])).astype(np.float32)
            else:
                raise RuntimeError("Requested 4 input channels but no DEM found.")
            # normalize DEM to [0,1]
            dem_tensor = torch.from_numpy(dem_np).float()
            dem_tensor = (dem_tensor - dem_tensor.min()) / (dem_tensor.max() - dem_tensor.min() + 1e-8)
            img_tensor = torch.cat((img_tensor, dem_tensor.unsqueeze(2)), axis=2)

        # reformat to C,H,W and normalize image to [0,1]
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0

        mask_tensor = torch.from_numpy(gt_np).unsqueeze(0).float() / 255.0

        # random flips
        img_tensor, mask_tensor = _apply_random_flips(img_tensor, mask_tensor)

        # photometric augmentations
        if self.transform:
            img_tensor = _apply_random_photometric_augmentations(img_tensor)

        return img_tensor.float(), mask_tensor.float()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = index % len(self.images)
        return self._load_pair(idx)


# -------------------------
# Concrete dataset wrappers
# -------------------------
def _read_list_file(list_path: Path) -> List[str]:
    """Read non-empty lines from a list file and return them as strings."""
    with list_path.open("r") as f:
        return [ln.strip() for ln in f if ln.strip()]


class OVAS(BaseCrackDataset):
    """OVAS dataset wrapper. Expects directory structure: <root>/<subset>/{image,gt,dem}."""

    def __init__(
        self,
        subset: str,
        list_file: Optional[str] = "list.txt",
        topo: bool = False,
        transform: bool = False,
        expand: bool = True,
        dilate: bool = True,
        in_channels: int = 3,
    ):
        root = Path("data/ovaskainen23_") / subset
        ext_img = "png"
        ext_gt = "tif"

        names = []
        if list_file:
            names = _read_list_file(root / list_file)

            images = [
                (root / "image" / n).with_suffix("." + ext_img)
                for n in names
                if n.endswith("." + ext_gt)
            ]
            masks = [root / "gt" / n for n in names if n.endswith("." + ext_gt)]
            dems = [root / "dem" / n for n in names if n.endswith("." + ext_gt)]
        else:
            images = sorted(path for path in (root / "image").iterdir() if path.suffix.lower().lstrip(".") == ext_img)
            masks = sorted(path for path in (root / "gt").iterdir() if path.suffix.lower().lstrip(".") == ext_gt)
            dems = sorted(path for path in (root / "dem").iterdir() if path.suffix.lower().lstrip(".") == ext_gt)

        super().__init__(images=images, masks=masks, dem_paths=dems, topo=topo, transform=transform,
                         expand=expand, dilate=dilate, in_channels=in_channels)


class MATTEO(BaseCrackDataset):
    """MATTEO dataset wrapper. Expects .tif files; includes DEM channel inside the image."""

    def __init__(
        self,
        subset: str,
        list_file: Optional[str] = "list.txt",
        topo: bool = False,
        transform: bool = False,
        expand: bool = True,
        dilate: bool = True,
        in_channels: int = 3,
    ):
        root = Path("data/matteo21") / subset
        ext = "tif"

        if list_file:
            names = _read_list_file(root / list_file)
        else:
            names = [p.name for p in (root / "image").iterdir() if p.suffix.lstrip(".") == ext]

        images = sorted(root / "image" / name for name in names)
        masks = sorted(root / "gt" / name for name in names)

        super().__init__(images=images, masks=masks, dem_paths=None, topo=topo, transform=transform,
                         expand=expand, dilate=dilate, in_channels=in_channels)


class SAMSU(BaseCrackDataset):
    """SAMSU dataset wrapper. Similar layout to OVAS."""

    def __init__(
        self,
        subset: str,
        list_file: Optional[str] = "list.txt",
        topo: bool = False,
        transform: bool = False,
        expand: bool = True,
        dilate: bool = True,
        in_channels: int = 3,
    ):
        root = Path("data/samsu19") / subset
        ext_img = "png"
        ext_gt = "tif"

        names = []
        if list_file:
            names = _read_list_file(root / list_file)
            images = [
                (root / "image" / n).with_suffix("." + ext_img)
                for n in names
                if n.endswith("." + ext_gt)
            ]
            masks = [root / "gt" / n for n in names if n.endswith("." + ext_gt)]
            dems = [root / "dem" / n for n in names if n.endswith("." + ext_gt)]
        else:
            images = sorted(p for p in (root / "image").iterdir() if p.suffix.lstrip(".") == ext_img)
            masks = sorted(p for p in (root / "gt").iterdir() if p.suffix.lstrip(".") == ext_gt)
            dems = sorted(p for p in (root / "dem").iterdir() if p.suffix.lstrip(".") == ext_gt)

        super().__init__(images=images, masks=masks, dem_paths=dems, topo=topo, transform=transform,
                         expand=expand, dilate=dilate, in_channels=in_channels)


class GeoCrack(BaseCrackDataset):
    """GeoCrack dataset wrapper (simple PNG images)."""

    def __init__(
        self,
        subset: str,
        topo: bool = False,
        transform: bool = False,
        expand: bool = True,
        dilate: bool = True,
        in_channels: int = 3,
    ):
        root = Path("data/GeoCrack_") / subset
        ext = "png"

        images = sorted(p for p in (root / "image").iterdir() if p.suffix.lstrip(".") == ext)
        masks = sorted(p for p in (root / "gt").iterdir() if p.suffix.lstrip(".") == ext)

        super().__init__(images=images, masks=masks, dem_paths=None, topo=topo, transform=transform,
                         expand=expand, dilate=dilate, in_channels=in_channels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = super().__getitem__(index)
        # consistent resizing used originally
        img = t.Resize(256)(img)
        mask = t.Resize(256)(mask)
        return img.float(), mask.float()


class DIC(BaseCrackDataset):
    """DIC dataset wrapper: single-channel images and PNG masks."""

    def __init__(
        self,
        subset: str,
        topo: bool = False,
        transform: bool = False,
        expand: bool = False,
        dilate: bool = False,
        in_channels: int = 1,
    ):
        root = Path("data/DIC") / subset
        ext_img = "tif"
        ext_mask = "png"

        images = sorted(p for p in (root / "image").iterdir() if p.suffix.lstrip(".") == ext_img)
        masks = sorted(p for p in (root / "gt").iterdir() if p.suffix.lstrip(".") == ext_mask)

        super().__init__(images=images, masks=masks, dem_paths=None, topo=topo, transform=transform,
                         expand=expand, dilate=dilate, in_channels=in_channels)

    def _load_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override to handle single-channel image format (the base expects >=3 channels).
        """
        img_np = _read_image(Path(self.images[idx]))
        gt_np = _read_mask(Path(self.masks[idx]))

        # ensure single channel
        if img_np.ndim == 3:
            img_np = img_np[..., 0]

        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(gt_np).unsqueeze(0).float() / 255.0

        img_tensor, mask_tensor = _apply_random_flips(img_tensor, mask_tensor)

        if self.transform:
            img_tensor = _apply_random_photometric_augmentations(img_tensor)

        img_tensor = t.Resize(256)(img_tensor)
        mask_tensor = t.Resize(256)(mask_tensor)

        return img_tensor.float(), mask_tensor.float()


# -------------------------
# Dataset registry & loader builder
# -------------------------
DATASETS = {
    "ovaskainen23": OVAS,
    "matteo21": MATTEO,
    "samsu19": SAMSU,
    "geocrack": GeoCrack,
    "dic": DIC,
}


def all_datasets(
    batch_size: int = 32,
    datasets: str = "samsu19-matteo21-ovaskainen23",
    in_channels: int = 4,
    out_channels: int = 1,
    shape: int = 256,
    expand: bool = True,
    dilate: bool = True,
    shuffle_train: bool = True,
    do_transform: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create concatenated train/val/test DataLoaders from multiple dataset names.

    Args:
        batch_size: batch size for DataLoaders.
        datasets: dash-separated dataset keys from DATASETS dict.
        in_channels: number of input channels requested (3 or 4).
        out_channels: number of output channels (kept for API compatibility).
        shape: target shape (not used directly here; datasets may resize internally).
        expand, dilate: whether to apply expand/dilate preprocessing.
        shuffle_train: whether to shuffle the training DataLoader.
        do_transform: whether to enable augmentations.

    Returns:
        Tuple(train_loader, val_loader, test_loader)
    """
    keys = [k.strip() for k in datasets.split("-") if k.strip()]
    all_train = []
    all_val = []
    all_test = []

    for name in keys:
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset key: {name}")
        DS = DATASETS[name]
        all_train.append(DS(subset="train", transform=do_transform, expand=expand, dilate=dilate, in_channels=in_channels))
        all_val.append(DS(subset="valid", transform=False, expand=expand, dilate=dilate, in_channels=in_channels))
        all_test.append(DS(subset="test", transform=False, expand=expand, dilate=dilate, in_channels=in_channels))

    trainset = ConcatDataset(all_train)
    valset = ConcatDataset(all_val)
    testset = ConcatDataset(all_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader
