"""Training-only manifest handling and image transforms."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms


SOURCES = {
    "BCN20000": "bcn20000_metadata.csv",
    "MSK-1": "msk-1_metadata.csv",
    "MSK-2": "msk-2_metadata.csv",
}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _group_lookup(data_root: Path) -> dict[tuple[str, str], str]:
    """Use a patient when available, then a lesion, then an image ID."""
    groups = {}
    for source, filename in SOURCES.items():
        path = data_root / "metadata" / "original" / filename
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                image_id = row["isic_id"]
                patient = row.get("patient_id", "").strip()
                lesion = row.get("lesion_id", "").strip()
                if patient:
                    group = f"{source}:patient:{patient}"
                elif lesion:
                    group = f"{source}:lesion:{lesion}"
                else:
                    group = f"{source}:image:{image_id}"
                groups[(source, image_id)] = group
    return groups


def make_split(data_root: Path, seed: int = 42, fold: int = 0) -> pd.DataFrame:
    """Create a source-and-label stratified, group-disjoint 80/20 split."""
    data_root = data_root.resolve()
    frame = pd.read_csv(data_root / "train.csv", dtype={"ID": str, "source": str})
    if list(frame.columns) != ["ID", "target", "source"]:
        raise ValueError("train.csv must have columns ID,target,source")
    if frame["ID"].duplicated().any():
        raise ValueError("Duplicate training IDs")
    if frame["target"].isna().any() or not set(frame["target"]).issubset({0, 1}):
        raise ValueError("Training targets must be 0 or 1")
    if not set(frame["source"]).issubset(SOURCES):
        raise ValueError("Unexpected training source")

    groups = _group_lookup(data_root)
    frame["group_id"] = [groups.get((s, i), f"{s}:image:{i}") for s, i in zip(frame.source, frame.ID)]
    frame["image_path"] = [str(data_root / "train_images" / s / f"{i}.jpg") for s, i in zip(frame.source, frame.ID)]
    missing = frame.loc[~frame["image_path"].map(lambda p: Path(p).is_file()), "ID"]
    if len(missing):
        raise FileNotFoundError(f"Missing training images: {missing.head(5).tolist()}")
    if not 0 <= fold < 5:
        raise ValueError("fold must be 0 through 4")

    strata = frame["source"] + ":" + frame["target"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, val_index = list(splitter.split(frame, strata, frame["group_id"]))[fold]
    frame["split"] = "train"
    frame.loc[val_index, "split"] = "val"
    train_groups = set(frame.loc[train_index, "group_id"])
    val_groups = set(frame.loc[val_index, "group_id"])
    if train_groups & val_groups:
        raise AssertionError("Train/validation group overlap")
    return frame


def image_transform(training: bool):
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.transform = image_transform(training)
        self.training = training

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.training or "target" in self.frame:
            return tensor, float(row.target)
        return tensor, str(row.ID)
