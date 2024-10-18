from os.path import join
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


def build_data(data_cfg):
    if data_cfg["data_type"] == "eg1800":
        root_dir = data_cfg["root_dir"]
        image_dir = data_cfg["image_dir"]
        label_dir = data_cfg["label_dir"]
        train_split = data_cfg["train_split"]
        test_split = data_cfg["test_split"]
        batch_size = data_cfg["batch_size"]

        trans, mask_trans = default_transform(data_cfg)

        train_set = SegmentationDataset(
            root_dir, image_dir, label_dir, train_split, trans, mask_trans
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = SegmentationDataset(
            root_dir, image_dir, label_dir, test_split, trans, mask_trans
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    elif data_cfg["data_type"] == "matting_human":
        root_dir = data_cfg["root_dir"]
        image_dir = data_cfg["image_dir"]
        label_dir = data_cfg["label_dir"]
        train_range = data_cfg["train_range"]
        test_range = data_cfg["test_range"]
        batch_size = data_cfg["batch_size"]

        trans, mask_trans = default_transform(data_cfg)

        train_set = MattingHuman(
            root_dir, image_dir, label_dir, train_range, trans, mask_trans
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = MattingHuman(
            root_dir, image_dir, label_dir, test_range, trans, mask_trans
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif data_cfg["data_type"] == "easy_portrait":
        root_dir = data_cfg["root_dir"]
        image_dir = data_cfg["image_dir"]
        label_dir = data_cfg["label_dir"]
        train_part = data_cfg["train_part"]
        test_part = data_cfg["test_part"]
        batch_size = data_cfg["batch_size"]

        trans, mask_trans = default_transform(data_cfg)

        train_set = EasyPortrait(
            root_dir, image_dir, label_dir, train_part, trans, mask_trans
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = EasyPortrait(
            root_dir, image_dir, label_dir, test_part, trans, mask_trans
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class EasyPortrait(Dataset):
    def __init__(
        self,
        root_dir,
        image_dir,
        label_dir,
        part="train",
        transform=None,
        mask_transform=None,
    ):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.part = part
        self.transform = transform
        self.mask_transform = mask_transform
        self._load_image_names()

    def _load_image_names(self):
        image_dir = Path(join(self.root_dir, self.image_dir, self.part))
        label_dir = Path(join(self.root_dir, self.label_dir, self.part))
        self.images = sorted(list(image_dir.glob("*jpg")))
        self.labels = sorted(list(label_dir.glob("*png")))

        print(f"Images: {len(self.images)}, Labels: {len(self.labels)}")
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.labels[idx]

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask


class MattingHuman(Dataset):
    def __init__(
        self,
        root_dir,
        image_dir,
        label_dir,
        data_range=(0, 0.7),
        transform=None,
        mask_transform=None,
    ):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.data_range = data_range
        self.transform = transform
        self.mask_transform = mask_transform
        self._load_image_names()
        self.start_idx = int(data_range[0] * len(self.images))

    def _load_image_names(self):
        root_dir = Path(self.root_dir)
        self.images = sorted(list(root_dir.glob(f"{self.image_dir}/*/*/*jpg")))
        self.labels = sorted(list(root_dir.glob(f"{self.label_dir}/*/*/*png")))

        self.images = [
            image
            for image in self.images
            if not "1803241125-00000005" in image.name
            and not "1803201916-00000117" in image.name
        ]
        self.labels = [
            label
            for label in self.labels
            if not "1803241125-00000005" in label.name
            and not "1803201916-00000117" in label.name
        ]

        print(f"Images: {len(self.images)}, Labels: {len(self.labels)}")
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return int(len(self.images) * (self.data_range[1] - self.data_range[0]))

    def __getitem__(self, idx):
        img_path = self.images[self.start_idx + idx]
        mask_path = self.labels[self.start_idx + idx]

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask


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

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask


class BinaryMaskToTensor:
    def __call__(self, mask):
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        return mask_tensor


class AlphaMaskToTensor:
    def __call__(self, mask):
        mask_tensor = torch.tensor(
            np.array(mask)[:, :, 3] == 0, dtype=torch.float32
        ).unsqueeze(0)
        return mask_tensor


class MultiMaskToTensor:
    def __call__(self, mask):
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        return mask_tensor


mask_pipeline = {
    "eg1800": BinaryMaskToTensor,
    "matting_human": AlphaMaskToTensor,
    "easy_portrait": MultiMaskToTensor,
}


def default_transform(data_cfg):
    input_size = data_cfg["input_size"]

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
            mask_pipeline[data_cfg["data_type"]](),
        ]
    )

    return transform, mask_transform


if __name__ == "__main__":
    train_loader, test_loader = build_data(
        {
            "data_type": "matting_human",
            "root_dir": "/kaggle/input/aisegmentcom-matting-human-datasets/",
            "image_dir": "clip_img",
            "label_dir": "matting",
            "train_split": "eg1800_train.txt",
            "test_split": "eg1800_test.txt",
            "train_range": (0, 0.5),
            "test_range": (0.5, 0.6),
            "input_size": (224, 224),
            "batch_size": 16,
        }
    )

    image_batch, mask_batch = next(iter(train_loader))
    print(image_batch.shape, image_batch.dtype)
    print(mask_batch.shape, mask_batch.dtype)
    print(image_batch.mean())
    print(image_batch.std())
    print(mask_batch.max())
    print(mask_batch.min())
