"""
Convert 4-band junction disk images into YOLO v8/v9/v11 format labels.
Each channel (0 to 3) corresponds to a class ID.

YOLO format per line:
class_id x_center y_center width height (all normalized between 0.0 and 1.0)
"""

import os
import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from pathlib import Path

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
THRESHOLD = 127                 # Threshold to binarize disks (e.g., 0.5 if float, 127 if uint8)

def process_4band_image(image_path, output_dir, threshold=THRESHOLD):
    # Load the multi-band image
    # Note: image shape will be (H, W, 4)
    img = imread(image_path)
    h, w, num_channels = img.shape
    
    if num_channels != 4:
        print(f"Warning: {image_path.name} has {num_channels} channels instead of 4. Skipping.")
        return

    yolo_lines = []

    # Iterate through each of the 4 bands (each band represents one class)
    for class_id in range(4):
        band = img[..., class_id]
        
        # Binarize the band to isolate the disks
        binary_mask = band > threshold
        
        # Label connected components (the individual disks)
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        
        for prop in regions:
            # bbox coordinates: (min_row, min_col, max_row, max_col)
            # which translates to: (ymin, xmin, ymax, xmax)
            ymin, xmin, ymax, xmax = prop.bbox
            
            # Calculate absolute bounding box dimensions
            box_width = xmax - xmin
            box_height = ymax - ymin
            
            # Calculate box center positions
            x_center = xmin + (box_width / 2.0)
            y_center = ymin + (box_height / 2.0)
            
            # Normalize coordinates relative to image dimensions (YOLO requirement)
            x_center_norm = x_center / w
            y_center_norm = y_center / h
            width_norm = box_width / w
            height_norm = box_height / h
            
            # Append the YOLO formatted string
            yolo_lines.append(
                f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            )

    # Determine output path (.txt extension with same base filename)
    image_stem = Path(image_path).stem
    txt_output_path = Path(output_dir) / f"{image_stem}.txt"
    
    # Save to a text file (creates an empty file if no junctions/disks are found)
    with open(txt_output_path, "w") as f:
        if yolo_lines:
            f.write("\n".join(yolo_lines) + "\n")
        else:
            f.write("")

def pipeline(input_folder, output_folder):
    in_path = Path(input_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Search for common image formats (adjust extensions if your 4-band data is in .tif/.tiff format)
    supported_extensions = ("*.png", "*.tif", "*.tiff")
    image_files = []
    for ext in supported_extensions:
        image_files.extend(in_path.glob(ext))
        
    if not image_files:
        print(f"No images found in {input_folder}. Please check your path and extensions.")
        return

    print(f"Found {len(image_files)} images to process.")
    for img_p in sorted(image_files):
        process_4band_image(img_p, out_path)
    print(f"Processing finished! YOLO labels saved to: {output_folder}")


INPUT_DIR = './data/ovaskainen23_/train/nodes'
OUTPUT_DIR = "./data/ovaskainen23_/train/yolo_labels"    # Folder where YOLO .txt files will be saved
pipeline(INPUT_DIR, OUTPUT_DIR)