import argparse
import numpy as np
import torch
from patchify import patchify, unpatchify
from PIL import Image
from skimage import io

from src.unet import UNet
from segmentation_models_pytorch import Segformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size = 256


# Model loader
def load_model(model_name: str):
    if model_name.lower() == "unet":
        model = UNet(init_features=64)
        weight_path = "model/unet.pt"
    elif model_name.lower() == "segformer":
        model = Segformer(
            encoder_name='resnet34',
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_segmentation_channels=256,
            in_channels=4,  # RGB + DEM
            classes=1,
            activation='sigmoid'
        )
        weight_path = "model/segformer.pt"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded {model_name} model from {weight_path}")
    return model


# Patch-based inference
def run_inference(model, img_arr, dem_arr):
    # Ensure RGB
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr]*3, axis=-1)
    if img_arr.shape[-1] > 3:
        img_arr = img_arr[:, :, :3]

    # Normalize DEM
    dem = (dem_arr - dem_arr.mean()) / (dem_arr.std() + 1e-8)

    # Merge RGB + DEM → 4 channels
    combined = np.concatenate((img_arr[:, :, :3], dem[..., None]), axis=-1)

    H, W, C = combined.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    combined_padded = np.pad(
        combined,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant"
    )

    # Patchify
    patches = patchify(combined_padded, (patch_size, patch_size, C), step=patch_size)

    pred_patches = []
    n_h, n_w, _, _, _ = patches.shape

    for i in range(n_h):
        for j in range(n_w):
            patch = patches[i, j]
            patch_t = torch.tensor(patch).float()
            patch_t = patch_t.permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                pred = model(patch_t.to(device)).cpu().numpy()
            pred_patches.append(pred)

    # Reconstruct full image
    pred = np.array(pred_patches).reshape(n_h, n_w, 1, patch_size, patch_size, 1)
    pred_full = unpatchify(pred, combined_padded.shape[:2] + (1,))
    pred_full = pred_full[:H, :W, 0]

    pred_mask = (pred_full * 255).astype(np.uint8)
    return pred_mask

# Run directly from here
# mask = run_fracture_inference(
#     image_path="test_rgb.png",
#     dem_path="test_dem.tif",
#     model_name="unet",
#     output_path="pred_mask.png"
# )

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Fracture segmentation inference")
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--dem", type=str, required=True, help="Path to DEM (.tif)")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "segformer"], help="Model name")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output path for predicted mask")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Read inputs
    img_arr = np.array(Image.open(args.image).convert("RGB"))
    dem_arr = io.imread(args.dem)

    # Run inference
    pred_mask = run_inference(model, img_arr, dem_arr)

    # Save output
    Image.fromarray(pred_mask).save(args.output)
    print(f"Saved prediction mask to {args.output}")


if __name__ == "__main__":
    main()
