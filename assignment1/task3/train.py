import os

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from portraitnet import build_model
from dataset import build_data
from utils import save_checkpoint, parse_args, load_config, Iou


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
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

        global_step = (epoch - 1) * len(train_loader) + i
        writer.add_scalar("Training/Loss", loss.item(), global_step)


def validate(model, test_loader, loss_f, metric_f, device, epoch, writer):
    model.eval()
    val_loss = 0.0
    val_metric = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_f(outputs, masks)
            metric = metric_f(outputs, masks)

            val_loss += loss.item()
            val_metric += metric.item()

            print(
                f"Epoch: [{epoch}], Val Iter: [{i}/{len(test_loader)}], Loss: {loss.item():.4f}, Iou: {metric.item():.4f}"
            )

    val_loss = val_loss / len(test_loader)
    val_metric = val_metric / len(test_loader)

    writer.add_scalar("Validation/Loss", val_loss, epoch)
    writer.add_scalar("Validation/IoU", val_metric, epoch)

    return val_loss, val_metric


def train():
    args = parse_args()
    config = load_config(args.config)

    # work_dir
    os.makedirs(config["work_dir"], exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config["work_dir"], "logs"))

    # device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # model
    model = build_model(config["model"])
    model = model.to(device)

    # data
    train_loader, test_loader = build_data(config["data"])

    # loss
    loss_f = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # metric
    metric_f = Iou()

    # Train and validate
    best_loss = float("inf")
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        train_one_epoch(model, train_loader, loss_f, optimizer, device, epoch, writer)

        # Validate
        if epoch % config["training"]["val_epochs"] == 0:
            val_loss, val_metric = validate(
                model, test_loader, loss_f, metric_f, device, epoch, writer
            )

            save_checkpoint(
                model, optimizer, config["work_dir"], epoch, val_loss < best_loss
            )
            print(f"Model saved at epoch {epoch}")

            if val_loss < best_loss:
                best_loss = val_loss
            print(f"Validation Loss: {val_loss:.4f}, Validation IoU: {val_metric:.4f}")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()
