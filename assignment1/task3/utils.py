# utils.py

# This file contains utility functions for the project.
# This file should include:
# - Loss computation (e.g., custom loss functions)
# - Metrics computation (e.g., IoU, mIoU)
# - Logging and visualization tools

import os

import torch


def save_checkpoint(model, optimizer, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path,
    )
