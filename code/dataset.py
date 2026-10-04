"""DeepWeeds CSV splits, transforms, and lazy PyTorch data loading."""
from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {"Filename", "Label", "Species"}
SUBSET_COLUMNS = {"Filename", "Label"}


def _require_columns(df: pd.DataFrame, source: str,
                     required: set[str] = REQUIRED_COLUMNS) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source}: thiếu cột CSV: {', '.join(sorted(missing))}")


def _labels(df: pd.DataFrame, source: str,
            required: set[str] = REQUIRED_COLUMNS) -> pd.Series:
    _require_columns(df, source, required)
    values = pd.to_numeric(df["Label"], errors="coerce")
    valid = values.notna() & (values % 1 == 0) & values.between(0, NUM_CLASSES - 1)
    if not valid.all():
        row = int(np.flatnonzero(~valid.to_numpy())[0]) + 2
        raise ValueError(f"{source}: Label không hợp lệ ở dòng CSV {row}; cần số nguyên 0..8")
    return values.astype(int)


def _filenames(df: pd.DataFrame, source: str,
               required: set[str] = REQUIRED_COLUMNS) -> pd.Series:
    _require_columns(df, source, required)
    names = df["Filename"]
    invalid = names.isna() | ~names.map(lambda value: isinstance(value, str) and bool(value.strip()))
    if invalid.any():
        row = int(np.flatnonzero(invalid.to_numpy())[0]) + 2
        raise ValueError(f"{source}: Filename trống hoặc không hợp lệ ở dòng CSV {row}")
    duplicate = names[names.duplicated()]
    if not duplicate.empty:
        raise ValueError(f"{source}: Filename trùng lặp: {duplicate.iloc[0]}")
    return names


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"{path}: cannot read CSV: {exc}") from exc


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read official metadata and three author-defined subsets in their CSV order."""
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("fold phải là số nguyên từ 0 đến 4")
    root = Path(labels_dir)
    metadata_path = root / "labels.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")
    metadata = _read_csv(metadata_path)
    _require_columns(metadata, str(metadata_path))
    _filenames(metadata, str(metadata_path))
    metadata = metadata.assign(Label=_labels(metadata, str(metadata_path)))
    invalid_species = metadata["Species"].isna() | ~metadata["Species"].map(
        lambda value: isinstance(value, str) and bool(value.strip()))
    if invalid_species.any():
        row = int(np.flatnonzero(invalid_species.to_numpy())[0]) + 2
        raise ValueError(f"{metadata_path}: Species empty or invalid at CSV row {row}")
    species_counts = metadata.groupby("Label")["Species"].nunique()
    inconsistent = species_counts[species_counts != 1]
    if not inconsistent.empty:
        label = int(inconsistent.index[0])
        raise ValueError(f"{metadata_path}: Label={label} maps to multiple Species")
    species_by_label = metadata.drop_duplicates("Label").set_index("Label")["Species"]
    metadata = metadata.set_index("Filename")
    frames = []
    for split in ("train", "val", "test"):
        path = root / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu file split: {path}")
        frame = _read_csv(path)
        _require_columns(frame, str(path), SUBSET_COLUMNS)
        _filenames(frame, str(path), SUBSET_COLUMNS)
        frame = frame.assign(Label=_labels(frame, str(path), SUBSET_COLUMNS))
        missing = ~frame["Filename"].isin(metadata.index)
        if missing.any():
            filename = frame.loc[missing, "Filename"].iloc[0]
            raise ValueError(f"{path}: Filename missing from {metadata_path.name}: {filename}")
        metadata_labels = frame["Filename"].map(metadata["Label"])
        label_mismatch = frame["Label"] != metadata_labels
        species = frame["Label"].map(species_by_label)
        if species.isna().any():
            label = int(frame.loc[species.isna(), "Label"].iloc[0])
            raise ValueError(f"{path}: no Species mapping in {metadata_path.name} for Label={label}")
        if "Species" in frame:
            invalid_species = frame["Species"].isna() | ~frame["Species"].map(
                lambda value: isinstance(value, str) and bool(value.strip()))
            if invalid_species.any():
                filename = frame.loc[invalid_species, "Filename"].iloc[0]
                raise ValueError(f"{path}: Species empty or invalid for Filename={filename}")
            species_mismatch = frame["Species"] != species
            if species_mismatch.any():
                filename = frame.loc[species_mismatch, "Filename"].iloc[0]
                raise ValueError(f"{path}: Species differs from Label class mapping for Filename={filename}")
        if label_mismatch.any():
            filename = frame.loc[label_mismatch, "Filename"].iloc[0]
            warnings.warn(
                f"{path}: {int(label_mismatch.sum())} row(s) have Label different from "
                f"{metadata_path.name}; first Filename={filename}",
                RuntimeWarning, stacklevel=2,
            )
        frame = frame.assign(Species=species)
        frames.append(frame.loc[:, ["Filename", "Label", "Species"]])
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None, expected_total: int | None = None) -> dict:
    """Validate names, labels and images; report counts without changing the split.

    Pass expected_total=17509 for the full official dataset; omit for synthetic data.
    """
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {}
    labels = {}
    for split, df in frames.items():
        names[split] = set(_filenames(df, split))
        labels[split] = _labels(df, split)
    overlap = {
        "train_val": names["train"] & names["val"],
        "train_test": names["train"] & names["test"],
        "val_test": names["val"] & names["test"],
    }
    for pair, duplicates in overlap.items():
        if duplicates:
            raise ValueError(f"Filename trùng giữa {pair.replace('_', ' và ')}: {sorted(duplicates)[0]}")
    total = sum(len(df) for df in frames.values())
    union = set().union(*names.values())
    if len(union) != total:
        raise ValueError(f"Filename trùng trong hợp ba tập: {total} dòng, {len(union)} tên duy nhất")
    if expected_total is not None and len(union) != expected_total:
        raise ValueError(f"Hợp ba tập có {len(union)} ảnh; cần {expected_total} ảnh")
    if images_dir is not None:
        root = Path(images_dir)
        for split, df in frames.items():
            for filename in df["Filename"]:
                if not (root / filename).is_file():
                    raise FileNotFoundError(f"{split}: thiếu ảnh Filename={filename} trong {root}")
    stats = {
        "n": {split: len(df) for split, df in frames.items()},
        "per_class": {
            split: {label: int((values == label).sum()) for label in range(NUM_CLASSES)}
            for split, values in labels.items()
        },
        "overlap": {pair: len(items) for pair, items in overlap.items()},
        "union": len(union),
    }
    print(stats)
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build ImageNet normalized transforms; evaluation is deterministic."""
    if not isinstance(img_size, int) or isinstance(img_size, bool) or img_size <= 0:
        raise ValueError("img_size phải là số nguyên dương")
    choices = {"basic", "color", "trivial", "randaug"}
    if aug not in choices:
        raise ValueError(f"aug không hợp lệ: {aug!r}; chọn một trong {sorted(choices)}")
    if train:
        steps = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
        if aug == "color":
            steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                saturation=0.2, hue=0.05))
        elif aug == "trivial":
            steps.append(transforms.TrivialAugmentWide())
        elif aug == "randaug":
            steps.append(transforms.RandAugment())
    else:
        steps = [transforms.Resize(256), transforms.CenterCrop(img_size)]
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


class DeepWeedsDataset(Dataset):
    """Load one RGB image per request; return (tensor, integer label, filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        _require_columns(df, "Dataset")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform if transform is not None else build_transforms(False)
        self.labels = _labels(self.df, "Dataset").tolist()
        _filenames(self.df, "Dataset")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        filename = self.df.iloc[i]["Filename"]
        path = self.images_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh Filename={filename}: {path}")
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                tensor = self.transform(rgb)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Không đọc được ảnh Filename={filename}: {path}") from exc
        return tensor, self.labels[i], filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                shuffle: bool | None = None, pin_memory: bool = True,
                seed: int | None = None, generator: torch.Generator | None = None):
    """Create a loader; balanced weights use only this training frame."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    if sampler not in (None, "balanced"):
        raise ValueError("sampler phải là None hoặc 'balanced'")
    if sampler == "balanced" and not train:
        raise ValueError("sampler='balanced' chỉ dùng cho train")
    if generator is not None and seed is not None:
        raise ValueError("Chỉ truyền một trong seed hoặc generator")
    if generator is None and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    dataset = DeepWeedsDataset(df, images_dir, transform)
    if shuffle is None:
        shuffle = train and sampler is None
    if sampler == "balanced" and shuffle:
        raise ValueError("Không thể dùng shuffle=True cùng sampler='balanced'")
    torch_sampler = None
    if sampler == "balanced":
        counts = pd.Series(dataset.labels).value_counts()
        weights = torch.as_tensor([1.0 / counts[label] for label in dataset.labels], dtype=torch.double)
        torch_sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                              generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if torch_sampler is None else False,
                      sampler=torch_sampler, num_workers=num_workers, pin_memory=pin_memory,
                      drop_last=bool(train and len(dataset) >= batch_size),
                      worker_init_fn=_seed_worker if seed is not None or generator is not None else None,
                      generator=generator)
