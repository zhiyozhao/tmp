from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


def build_data(data_cfg):
    root_dir = data_cfg["root_dir"]
    image_dir = data_cfg["image_dir"]
    label_dir = data_cfg["label_dir"]
    train_split = data_cfg["train_split"]
    test_split = data_cfg["test_split"]
    input_size = data_cfg["input_size"]
    batch_size = data_cfg["batch_size"]

    trans, mask_trans = default_transform(input_size)

    train_set = SegmentationDataset(
        root_dir, image_dir, label_dir, train_split, trans, mask_trans
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = SegmentationDataset(
        root_dir, image_dir, label_dir, test_split, trans, mask_trans
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_dir,
        label_dir,
        split_file,
        transform=None,
        mask_transform=None,
    ):
        self.root_dir = root_dir
        self.image_dir = join(root_dir, image_dir)
        self.label_dir = join(root_dir, label_dir)
        self.split_file = join(root_dir, split_file)
        self.transform = transform
        self.mask_transform = mask_transform
        self._load_image_names()

    def _load_image_names(self):
        with open(self.split_file, "r") as file:
            image_names = [line.strip() for line in file]

        self.images = image_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = join(self.image_dir, img_name)
        mask_path = join(self.label_dir, img_name)

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


if __name__ == "__main__":
    train_loader, test_loader = build_data(
        {
            "root_dir": "/Users/zhao/Downloads/EG1800",
            "image_dir": "Images",
            "label_dir": "Labels",
            "train_split": "eg1800_train.txt",
            "test_split": "eg1800_test.txt",
            "input_size": (224, 224),
            "batch_size": 32,
        }
    )

    image_batch, mask_batch = next(iter(train_loader))
    print(image_batch.shape, image_batch.dtype)
    print(mask_batch.shape, mask_batch.dtype)
    print(image_batch.mean())
    print(image_batch.std())
    print(mask_batch.max())
    print(mask_batch.min())
