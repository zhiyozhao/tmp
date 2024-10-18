from os.path import join
import argparse

import yaml
import torch


class Iou:
    def __call__(self, pred, target):
        """
        Input:
            pred: (bs, 1, h, w) logits
            target: (bs, 1, h, w) binary mask
        """

        pred = (pred.sigmoid() > 0.5).float()
        target = target.float()

        intersection = (pred * target).sum(dim=[2, 3])
        union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) - intersection

        iou = intersection / (union + 1e-8)
        iou = iou.mean()

        return iou


class mIou:
    def __call__(self, pred, target):
        """
        Input:
            pred: (bs, c, h, w) logits
            target: (bs, h, w) multi-class mask
        """
        pred_classes = torch.argmax(pred, dim=1)

        num_classes = pred.shape[1]
        iou_per_class = []

        for cls_ in range(num_classes):
            pred_is_cls = pred_classes == cls_
            target_is_cls = target == cls_

            intersection = torch.logical_and(pred_is_cls, target_is_cls).sum()
            union = torch.logical_or(pred_is_cls, target_is_cls).sum()

            iou = (intersection / (union + 1e-6)).item()
            iou_per_class.append(iou)

        mean_iou = sum(iou_per_class) / num_classes

        return torch.tensor(mean_iou)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a portrait segmentation model")
    parser.add_argument("--config", required=True, help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, work_dir, epoch, is_best=False):
    state = (
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
    )
    checkpoint_path = join(work_dir, f"epoch_{epoch}.pth")
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = join(work_dir, f"best.pth")
        torch.save(state, best_path)
