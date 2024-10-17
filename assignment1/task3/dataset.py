# dataset.py

# Define custom dataset classes and data loading functions.
# This file should include:
# 1. Custom Dataset class(es) inheriting from torch.utils.data.Dataset
# 2. Data loading and preprocessing functions
# 3. Data augmentation techniques (if applicable)
# 4. Functions to split data into train/val/test sets
# 5. Any necessary data transformations

# About the dataset directory structure:
# 1. Images/ contains all png images to be segmented
# 2. Labels/ contains all corresponding png masks with same name as the image
# 3. eg1800_train.txt train split, each line contains the name of the image
# 4. eg1800_test.txt test split, each line contains the name of the image

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(
        self, image_dir, label_dir, split_file, transform=None, mask_transform=None
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = self._load_image_names(split_file)

    def _load_image_names(self, split_file):
        with open(split_file, "r") as file:
            image_names = [line.strip() for line in file]
        return image_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.label_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask


class BinaryMaskToTensor:
    def __call__(self, mask):
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        return mask_tensor


def default_transform(input_size):

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                input_size, interpolation=transforms.InterpolationMode.NEAREST
            ),
            BinaryMaskToTensor(),
        ]
    )

    return transform, mask_transform


def get_data_loader(
    batch_size, image_dir, label_dir, split_file, transform=None, mask_transform=None
):
    dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        split_file=split_file,
        transform=transform,
        mask_transform=mask_transform,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    batch_size = 32
    image_dir = "/Users/zhao/Downloads/EG1800/Images"
    label_dir = "/Users/zhao/Downloads/EG1800/Labels"
    train_split_file = "/Users/zhao/Downloads/EG1800/eg1800_train.txt"
    test_split_file = "/Users/zhao/Downloads/EG1800/eg1800_test.txt"
    transform, mask_transform = default_transform(input_size=(224, 224))

    train_loader = get_data_loader(
        batch_size, image_dir, label_dir, train_split_file, transform, mask_transform
    )
    test_loader = get_data_loader(
        batch_size, image_dir, label_dir, test_split_file, transform, mask_transform
    )

    image_batch, mask_batch = next(iter(train_loader))
    print(image_batch.shape, image_batch.dtype)
    print(mask_batch.shape, mask_batch.dtype)
    print(image_batch.mean())
    print(image_batch.std())
    print(mask_batch.max())
    print(mask_batch.min())
