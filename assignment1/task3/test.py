import os
from os.path import join

import torch

from portraitnet import build_model
from dataset import build_data
from utils import parse_args, load_config, mIou


def validate(model, test_loader, metric_f, device):
    model.eval()
    val_metric = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            metric = metric_f(outputs, masks)

            val_metric += metric.item()

            print(f"Val Iter: [{i}/{len(test_loader)}], Iou: {metric.item():.4f}")

    val_metric = val_metric / len(test_loader)

    return val_metric


def test():
    args = parse_args()
    config = load_config(args.config)

    # work_dir
    os.makedirs(config["work_dir"], exist_ok=True)

    # device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # model
    model = build_model(config["model"])
    model = model.to(device)
    checkpoint_path = join(config["work_dir"], "best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # data
    _, test_loader = build_data(config["data"])

    # metric
    metric_f = mIou()

    # testing
    val_metric = validate(model, test_loader, metric_f, device)
    print(f"Validation IoU: {val_metric:.4f}")


if __name__ == "__main__":
    test()
