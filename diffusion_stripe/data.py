from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    key: str
    path: Path


@dataclass(frozen=True)
class PairedRecord:
    key: str
    original_path: Path
    striped_path: Path


def _iter_image_files(root: Path, splits: Iterable[str] = SPLITS) -> Iterable[Path]:
    selected = tuple(splits)
    if not selected or len(selected) != len(set(selected)) or any(split not in SPLITS for split in selected):
        raise ValueError("splits must be a nonempty, nonduplicate subset of train/val/test")
    for split in selected:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def collect_images(
    root: str | Path,
    expected_count: int | None = 6199,
    splits: Iterable[str] = SPLITS,
) -> list[ImageRecord]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    records = [
        ImageRecord(key=path.relative_to(root_path).as_posix(), path=path)
        for path in _iter_image_files(root_path, splits)
    ]
    records.sort(key=lambda item: item.key)

    keys = [item.key for item in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate relative image paths found under {root_path}")
    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} images under {root_path}, found {len(records)}"
        )
    return records


def collect_pairs(
    original_root: str | Path,
    striped_root: str | Path,
    expected_count: int | None = 6199,
    splits: Iterable[str] = SPLITS,
) -> list[PairedRecord]:
    selected = tuple(splits)
    originals = collect_images(original_root, expected_count=expected_count, splits=selected)
    stripes = collect_images(striped_root, expected_count=expected_count, splits=selected)
    original_by_key = {item.key: item.path for item in originals}
    stripe_by_key = {item.key: item.path for item in stripes}

    missing_stripes = sorted(set(original_by_key) - set(stripe_by_key))
    missing_originals = sorted(set(stripe_by_key) - set(original_by_key))
    if missing_stripes or missing_originals:
        details = []
        if missing_stripes:
            details.append(f"missing stripes (first 5): {missing_stripes[:5]}")
        if missing_originals:
            details.append(f"missing originals (first 5): {missing_originals[:5]}")
        raise RuntimeError("Original/stripe pairing failed: " + "; ".join(details))

    return [
        PairedRecord(
            key=key,
            original_path=original_by_key[key],
            striped_path=stripe_by_key[key],
        )
        for key in sorted(original_by_key)
    ]


def load_rgb_tensor(path: Path, image_size: int = 64) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            raise RuntimeError(
                f"Expected {image_size}x{image_size} image, got {image.size}: {path}"
            )
        return pil_to_tensor(image).float().div_(255.0)


class StripedImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        striped_root: str | Path,
        image_size: int = 64,
        expected_count: int | None = 6199,
    ) -> None:
        self.records = collect_images(striped_root, expected_count=expected_count)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_rgb_tensor(self.records[index].path, image_size=self.image_size)


class OriginalStripePairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        original_root: str | Path,
        striped_root: str | Path,
        image_size: int = 64,
        expected_count: int | None = 6199,
        splits: Iterable[str] = SPLITS,
    ) -> None:
        self.splits = tuple(splits)
        self.records = collect_pairs(
            original_root=original_root,
            striped_root=striped_root,
            expected_count=expected_count,
            splits=self.splits,
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        original = load_rgb_tensor(record.original_path, image_size=self.image_size)
        striped = load_rgb_tensor(record.striped_path, image_size=self.image_size)
        return original, striped
