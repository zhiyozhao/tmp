import numpy as np
import torch


class StereoMatchingNetwork(torch.nn.Module):
    """
    The network should consist of the following layers:
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - functional.normalize(..., dim=1, p=2)

    Remark: Note that the convolutional layers expect the data to have shape
        `batch size * channels * height * width`. Permute the input dimensions
        accordingly for the convolutions and remember to revert it before returning the features.
    """

    def __init__(self):
        """
        Implementation of the network architecture.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """

        super().__init__()
        gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 5)
        # -------------------------------------

    def forward(self, X):
        """
        The forward pass of the network. Returns the features for a given image patch.

        Args:
            X (torch.Tensor): image patch of shape (batch_size, height, width, n_channels)

        Returns:
            features (torch.Tensor): predicted normalized features of the input image patch X,
                               shape (batch_size, height - 8, width - 8, n_features)
        """

        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 5)
        # -------------------------------------


def calculate_similarity_score(infer_similarity_metric, Xl, Xr):
    """
    Computes the similarity score for two stereo image patches.

    Args:
        infer_similarity_metric (torch.nn.Module):  pytorch module object
        Xl (torch.Tensor): tensor holding the left image patch
        Xr (torch.Tensor): tensor holding the right image patch

    Returns:
        score (torch.Tensor): the similarity score of both image patches which is the dot product of their features
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5)
    # -------------------------------------
