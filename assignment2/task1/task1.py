import os.path as osp

import cv2
import numpy as np


def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter.
    Args:
        img: the input image to be Gaussian filtered.
        kernel_size: the kernel size in both the X and Y directions.
        sigma: the standard deviation in both the X and Y directions.
    Returns:
        res_img: the output image after Gaussian filter.
    """
    # TODO: implement the Gaussian filter function.
    # Placeholder that you can delete. An image with all zeros.
    res_img = np.zeros_like(img)

    return res_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Lena-RGB.jpg"))
    kernel_size = 5
    sigma = 1
    res_img = gaussian_filter(img, kernel_size, sigma)

    cv2.imwrite(osp.join(root_dir, "gaussian_result.jpg"), res_img)
