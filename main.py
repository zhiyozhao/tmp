import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor, AutoProcessor
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import the scheduler

from dataset import VQADataset, build_collate_fn
from clip import CLIPForVQA
from baseline import TaskSpecificVQA


# Load configuration from a json file
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


# Training loop
def train_epoch(
    model, dataloader, optimizer, scheduler, criterion, device, writer, epoch, config
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(
        dataloader
    ):
        # Move data to the device
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % config["log_interval"] == 0:
            print(
                f"Epoch [{epoch}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    writer.add_scalar("Training Loss", avg_loss, epoch)
    writer.add_scalar("Training Accuracy", accuracy, epoch)

    print(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Step the scheduler
    scheduler.step()


# Validation loop
def validate_epoch(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, input_ids, attention_mask, labels in dataloader:
            # Move data to the device
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Logging
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    writer.add_scalar("Validation Loss", avg_loss, epoch)
    writer.add_scalar("Validation Accuracy", accuracy, epoch)

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


# Main training and evaluation loop
def main(cfg):
    # Load the configuration settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model_name"]
    image_model_name = cfg["image_model_name"]
    text_model_name = cfg["text_model_name"]
    num_classes = cfg["num_classes"]
    fusion_dim = cfg["fusion_dim"]
    dropout_prob = cfg["dropout_prob"]
    batch_size = cfg["batch_size"]
    num_epochs = cfg["num_epochs"]
    learning_rate = cfg["learning_rate"]
    checkpoint_dir = cfg["checkpoint_dir"]
    log_dir = cfg["log_dir"]

    # Dataset and DataLoader
    train_dataset = VQADataset(
        cfg["data_root"], cfg["train_csv"], cfg["answer_space_file"]
    )
    val_dataset = VQADataset(cfg["data_root"], cfg["val_csv"], cfg["answer_space_file"])

    if model_name is not None:
        collate_fn = build_collate_fn(
            combined_processor=CLIPProcessor.from_pretrained(model_name)
        )
    else:
        image_processor = AutoProcessor.from_pretrained(image_model_name, use_fast=True)
        text_processor = AutoProcessor.from_pretrained(text_model_name, use_fast=True)
        collate_fn = build_collate_fn(
            image_processor=image_processor, text_processor=text_processor
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model, criterion, optimizer
    if model_name is not None:
        model = CLIPForVQA(
            model_name,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            dropout_prob=dropout_prob,
        ).to(device)
    else:
        model = TaskSpecificVQA(
            image_model_name,
            text_model_name,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            dropout_prob=dropout_prob,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )  # Initialize the scheduler

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        # Training
        train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            criterion,
            device,
            writer,
            epoch,
            cfg,
        )

        # Validation
        val_loss, val_accuracy = validate_epoch(
            model, val_dataloader, criterion, device, writer, epoch
        )

        # Save model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_model_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model checkpoint at epoch {epoch}.")

    print("Training complete.")


if __name__ == "__main__":
    # Load the configuration file
    config_path = "config.json"  # Path to your config file
    cfg = load_config(config_path)

    # Run the main loop
    main(cfg)
