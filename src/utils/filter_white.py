import os
import argparse
from PIL import Image


def is_mostly_white(image, threshold=.8):
    """
    Checks if the image is mostly white based on a threshold.

    Args:
    image: The PIL Image object.
    threshold: The minimum percentage of white pixels.

    Returns:
    True if the percentage of white pixels is above the threshold,
    False otherwise.
    """

    width, height = image.size
    pixels = image.load()
    count = 0

    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            if r == g == b == 255:
                count += 1

    percentage = count / (width * height)

    return percentage > threshold


def filter_non_white_images(image_folder):
    """
    Filters images in a folder and saves filenames of non-white ones to a file.

    Args:
        image_folder: Path to the folder containing images.
    """

    non_white_images = []

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if os.path.isfile(image_path):
            try:
                image = Image.open(image_path)
                if not is_mostly_white(image):
                    non_white_images.append(filename)
            except (IOError, OSError) as e:
                print(f"Error opening image: {filename} ({e})")

    with open(os.path.join(image_folder, 'list.txt'), 'w') as f:
        f.write('\n'.join(non_white_images))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter non-white images.")
    parser.add_argument("image_folder",
                        help="Path to the folder containing images.")
    args = parser.parse_args()

    filter_non_white_images(args.image_folder)
