import os
from os.path import join

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from portraitnet import build_model
from dataset import default_transform
from utils import parse_args, load_config


def visualize_inference(image, mask, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    ax[0].imshow(image)
    ax[0].axis("off")

    # Segmentation Mask
    ax[1].imshow(image)
    ax[1].imshow(mask, alpha=0.5, cmap="jet")
    ax[1].axis("off")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def infer(model, image_paths, device, transform, save_dir):
    model.eval()

    for img_path in image_paths:
        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        original_image = image.copy()

        # Preprocess image
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        with torch.no_grad():
            # Model inference
            output = model(image)

        # Convert output to binary mask
        mask = (output.sigmoid() > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()
        mask = Image.fromarray(
            (mask * 255).astype(np.uint8)
        )  # Convert to binary image (0, 255)
        mask = mask.resize(
            original_size, Image.NEAREST
        )  # Resize mask to original image size

        # Visualize and save the output
        filename = os.path.basename(img_path)
        save_path = join(save_dir, f"{filename}")
        visualize_inference(original_image, mask, save_path=save_path)


def infer():
    args = parse_args()
    config = load_config(args.config)

    # save_dir
    save_dir = join(config["work_dir"], "infer")
    os.makedirs(save_dir, exist_ok=True)

    # device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # model
    model = build_model(config["model"])
    model = model.to(device)
    checkpoint_path = join(config["work_dir"], "best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # transform
    input_size = config["data"]["input_size"]
    transform, _ = default_transform(config["data"])

    # Infer on images from a folder
    image_dir = config["infer"]["image_dir"]
    image_paths = [
        join(image_dir, img)
        for img in os.listdir(image_dir)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Run inference
    infer(model, image_paths, device, transform, input_size, save_dir)


if __name__ == "__main__":
    infer()
