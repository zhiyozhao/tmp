import os
import os.path as osp
import argparse

import yaml
import torch
from torch import nn
import torch.optim as optim

from portraitnet import PortraitNet
from dataset import get_data_loader, default_transform
from utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train a portrait segmentation model")
    parser.add_argument("--config", required=True, help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    model_cfg = config["model"]
    model = PortraitNet(
        backbone_type=model_cfg["backbone_type"], num_classes=model_cfg["num_classes"]
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def build_data(config):
    data_cfg = config["data"]

    batch_size = data_cfg["batch_size"]
    image_dir = osp.join(data_cfg["root_dir"], data_cfg["image_dir"])
    label_dir = osp.join(data_cfg["root_dir"], data_cfg["label_dir"])
    train_split = osp.join(data_cfg["root_dir"], data_cfg["train_split"])
    test_split = osp.join(data_cfg["root_dir"], data_cfg["test_split"])
    trans, mask_trans = default_transform(data_cfg["input_size"])
    train_loader = get_data_loader(
        batch_size,
        image_dir,
        label_dir,
        train_split,
        trans,
        mask_trans,
    )
    test_loader = get_data_loader(
        batch_size,
        image_dir,
        label_dir,
        test_split,
        trans,
        mask_trans,
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for i, (images, masks) in enumerate(train_loader, 1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(
            f"Epoch: [{epoch}], Iter: [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
        )

    return running_loss / len(train_loader)


def validate(model, test_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

            print(
                f"Epoch: [{epoch}], Val Iter: [{i}/{len(test_loader)}], Loss: {loss.item():.4f}"
            )

    return val_loss / len(test_loader)


def main():
    args = parse_args()
    config = load_config(args.config)

    # work_dir
    os.makedirs(config["work_dir"], exist_ok=True)

    # model
    model, device = build_model(config)

    # data
    train_loader, test_loader = build_data(config)

    # loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # Train and validate
    best_loss = float("inf")
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if epoch % config["training"]["val_epochs"] == 0:
            val_loss = validate(model, test_loader, criterion, device, epoch)
            print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint if the validation loss has improved
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, config["output"]["checkpoint_dir"], epoch)
            print(f"Model saved at epoch {epoch + 1}")


if __name__ == "__main__":
    main()
