import math
import os

from PIL import Image


def assemble_from_tiles(image_name, tile_dir, ext, output_path):
    """
    Stitches a collection of tiles back into the original image.

    Args:
    image_name: Name prefix of the original image (used in tile filenames).
    tile_dir: Directory containing the split image tiles.
    ext: File extension of the image tiles (e.g., "jpg", "png").
    output_path: Path to save the assembled image.
    """
    # Find all tile files
    tile_files = [f for f in os.listdir(tile_dir)
                  if f.startswith(image_name) and f.endswith(ext)]

    # Extract tile dimensions from the first filename
    first_tile_name = tile_files[0]
    _, row, col = first_tile_name.split("_")[1:]
    tile_size = 256

    # Determine the number of rows and columns based on tile count
    num_tiles = len(tile_files)
    num_cols = math.ceil(num_tiles**0.5)  # Assuming square grid of tiles
    num_rows = num_tiles // num_cols

    # Create a new image object with the calculated size
    assembled_image = Image.new("RGB",
                                (num_cols * tile_size, num_rows * tile_size))

    # Loop through each tile and paste it into the assembled image
    for i, tile_file in enumerate(tile_files):
        row = i // num_cols
        col = i % num_cols
        tile_path = os.path.join(tile_dir, tile_file)
        tile = Image.open(tile_path)
        assembled_image.paste(tile, (col * tile_size, row * tile_size))

    # Save the assembled image
    assembled_image.save(output_path)
    print(f"Image assembled from tiles and saved to: {output_path}")
