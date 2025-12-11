from ridge_detection.lineDetector import LineDetector
from ridge_detection.params import Params
from ridge_detection.helper import displayContours, save_to_disk
from PIL import Image
import numpy as np

DEFAULT_RIDGE_CONFIG = {
    # "path_to_file": img_path,
    "mandatory_parameters": {
        # "Sigma": 3.39,
        # "Lower_Threshold": 0.34,
        # "Upper_Threshold": 1.02,
        "Maximum_Line_Length": 0,
        "Minimum_Line_Length": 0,
        "Darkline": "LIGHT",
        "Overlap_resolution": "NONE",
    },
    "optional_parameters": {
        "Line_width": 3,
        "High_contrast": 200,
        "Low_contrast": 60,
    },
    "further_options": {
        "Correct_position": True,
        "Estimate_width": True,
        "doExtendLine": True,
        "Show_junction_points": True,
        "Show_IDs": False,
        "Display_results": False,
        "Preview": False,
        "Make_Binary": True,
        "save_on_disk": False,
    },
}


def img_ridge_coords_to_binmat(ridge_results,
                               shape=(256, 256), mat_per_line=False):
    if mat_per_line:
        to_return = list()
    else:
        to_return = np.zeros(shape=shape)
    for line in ridge_results:
        if mat_per_line:
            line_mat = np.zeros(shape=shape)
            for row_val, col_val in zip(line.row, line.col):
                line_mat[row_val, col_val] = 1
            to_return.append(line_mat)
        else:
            for row_val, col_val in zip(line.row, line.col):
                to_return[row_val, col_val] = 1

    return np.dstack([to_return, to_return, to_return])


def ridge_from_file(img_path):

    ridge_config = DEFAULT_RIDGE_CONFIG
    ridge_config["path_to_file"] = img_path
    params = Params(ridge_config)

    img = Image.open(ridge_config["path_to_file"])

    detect = LineDetector(params=params)
    result = detect.detectLines(img)
    resultJunction = detect.junctions

    out_img, img_only_lines = displayContours(params, result, resultJunction)

    if params.get_saveOnFile() is True:
        save_to_disk(out_img, img_only_lines, '.')

    just_edges = img_ridge_coords_to_binmat(
        result, img.size, mat_per_line=False)

    # io.imsave('__test__.png', (just_edges * 255).astype(np.uint8))
    return just_edges
