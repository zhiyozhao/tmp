import os
import os.path as osp

import numpy as np
import torch
from block_matching import add_padding, visualize_disparity
from dataset import KITTIDataset
from siamese_neural_network import StereoMatchingNetwork


def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50):
    """
    Computes the disparity of the stereo image pair.

    Args:
        infer_similarity_metric:  pytorch module object
        img_l: tensor holding the left image
        img_r: tensor holding the right image
        max_disparity (int): maximum disparity

    Returns:
        D: tensor holding the disparity
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 8)
    # -------------------------------------


def main():
    # Hyperparameters
    training_iterations = 1000
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network")
    model_path = osp.join(out_dir, f"trained_model_{training_iterations}_final.pth")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Set network to eval mode
    infer_similarity_metric = StereoMatchingNetwork()
    infer_similarity_metric.load_state_dict(torch.load(model_path))
    infer_similarity_metric.eval()
    infer_similarity_metric.to("cpu")

    # Load KITTI test split
    dataset = KITTIDataset(osp.join(data_dir, "testing"))
    # Loop over test images
    for i in range(len(dataset)):
        print(f"Processing {i} image")
        # Load images and add padding
        img_left, img_right = dataset[i]
        img_left_padded, img_right_padded = add_padding(img_left, padding), add_padding(
            img_right, padding
        )
        img_left_padded, img_right_padded = torch.Tensor(img_left_padded), torch.Tensor(
            img_right_padded
        )

        disparity_map = compute_disparity_CNN(
            infer_similarity_metric,
            img_left_padded,
            img_right_padded,
            max_disparity=max_disparity,
        )
        # Visulization
        title = (
            f"Disparity map for image {i} with SNN (training iterations {training_iterations}, "
            f"batch size {batch_size}, patch_size {patch_size})"
        )
        file_name = f"{i}_training_iterations_{training_iterations}.png"
        out_file_path = osp.join(out_dir, file_name)
        visualize_disparity(
            disparity_map.squeeze(),
            img_left.squeeze(),
            img_right.squeeze(),
            out_file_path,
            title,
            max_disparity=max_disparity,
        )


if __name__ == "__main__":
    main()
