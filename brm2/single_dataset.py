from pathlib import Path
from typing import TypedDict

import albumentations as A
import torch
from torch.utils.data import Dataset

from dataset_utils import parse_data_sources, read_image


class SingleDatasetSample(TypedDict):
    """Dictionary type for a single dataset sample."""

    image: torch.Tensor
    label: torch.Tensor
    atlas_name: str
    filename: str


class BrainwaysSingleDataset(Dataset):
    """Dataset for handling Brainways single dataset."""

    def __init__(
        self,
        data_sources: list[tuple[Path, str]],
        transform: A.TransformType | None = None,
    ):
        self._labels = parse_data_sources(data_sources)
        self._transform = transform

    def __getitem__(self, index) -> SingleDatasetSample:
        """Get the dataset sample at the given index."""
        if index >= len(self._labels):
            raise IndexError
        filename = self._labels.iloc[index].filename
        image = read_image(filename)
        if self._transform is not None:
            image = self._transform(image=image)["image"]
        ap_value = float(self._labels.iloc[index]["ap"])
        atlas_name = self._labels.iloc[index]["atlas_name"]
        return {
            "image": image,
            "ap": ap_value,
            "atlas_name": atlas_name,
            "filename": filename,
        }

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self._labels)
