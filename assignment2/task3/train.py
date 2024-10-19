import os
import os.path as osp

import numpy as np
import torch
from dataset import KITTIDataset, PatchProvider
from siamese_neural_network import StereoMatchingNetwork, calculate_similarity_score


def hinge_loss(score_pos, score_neg, label):
    """
    Computes the hinge loss for the similarity of a positive and a negative example.

    Args:
        score_pos (torch.Tensor): similarity score of the positive example
        score_neg (torch.Tensor): similarity score of the negative example
        label (torch.Tensor): the true labels

    Returns:
        avg_loss (torch.Tensor): the mean loss over the patch and the mini batch
        acc (torch.Tensor): the accuracy of the prediction
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 6)
    # -------------------------------------


def training_loop(
    infer_similarity_metric,
    patches,
    optimizer,
    out_dir,
    iterations=1000,
    batch_size=128,
):
    """
    Runs the training loop of the siamese network.

    Args:
        infer_similarity_metric (obj): pytorch module
        patches (obj): patch provider object
        optimizer (obj): optimizer object
        out_dir (str): output file directory
        iterations (int): number of iterations to perform
        batch_size (int): batch size
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 6)
    # -------------------------------------


def main():
    # Fix random seed for reproducibility
    np.random.seed(7)
    torch.manual_seed(7)

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
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Create dataloader for KITTI training set
    dataset = KITTIDataset(
        osp.join(data_dir, "training"),
        osp.join(data_dir, "training/disp_noc_0"),
    )
    # Load patch provider
    patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))

    # Initialize the network
    infer_similarity_metric = StereoMatchingNetwork()
    # Set to train
    infer_similarity_metric.train()
    # uncomment if you don't have a gpu
    # infer_similarity_metric.to('cpu')
    optimizer = torch.optim.SGD(
        infer_similarity_metric.parameters(), lr=learning_rate, momentum=0.9
    )

    # Start training loop
    training_loop(
        infer_similarity_metric,
        patches,
        optimizer,
        out_dir,
        iterations=training_iterations,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
