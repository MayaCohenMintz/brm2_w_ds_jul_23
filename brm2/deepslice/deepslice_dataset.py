from pathlib import Path
from typing import TypedDict

import albumentations as A
import torch
from torch.utils.data import Dataset

from dataset_utils import parse_data_sources, read_image


class DeepSliceDatasetSample(TypedDict):
    """Dictionary type for a single dataset sample."""

    image: torch.Tensor
    label: torch.Tensor
    atlas_name: str
    filename: str


class DeepSliceDataset(Dataset):
    """Dataset for handling Brainways single dataset."""

    def __init__(
        self,
        data_sources: list[tuple[Path, str]],
        transform: A.TransformType | None = None,
    ):
        self._labels = parse_data_sources(data_sources) 
        self._transform = transform

    def __getitem__(self, index) -> DeepSliceDatasetSample:
        """Get the dataset sample at the given index."""
        if index >= len(self._labels):
            raise IndexError
        filename = self._labels.iloc[index].filename
        # MAYA - this is a "plaster": added this since my DS gt alignments don't have extensions. This should be removed when running on different train/val/test datasets
        valid_suffixes = ['.png', '.jpeg', '.jpg']
        if not any(filename.lower().endswith(suffix) for suffix in valid_suffixes):
            filename += '.jpg'
        # end of addition
        image = read_image(filename) 
        if self._transform is not None:
            image = self._transform(image=image)["image"]
        ap_value = torch.as_tensor([float(self._labels.iloc[index]["ap"])]) 
        # MAYA - added this: 
        #alignment_vecs = torch.as_tensor([float(self._labels.iloc[index]["OUV_vecs"])])
        ox = torch.as_tensor([float(self._labels.iloc[index]["ox"])])
        oy = torch.as_tensor([float(self._labels.iloc[index]["oy"])])
        oz = torch.as_tensor([float(self._labels.iloc[index]["oz"])])
        ux = torch.as_tensor([float(self._labels.iloc[index]["ux"])])
        uy = torch.as_tensor([float(self._labels.iloc[index]["uy"])])
        uz = torch.as_tensor([float(self._labels.iloc[index]["uz"])])
        vx = torch.as_tensor([float(self._labels.iloc[index]["vx"])])
        vy = torch.as_tensor([float(self._labels.iloc[index]["vy"])])
        vz = torch.as_tensor([float(self._labels.iloc[index]["vz"])])
        atlas_name = self._labels.iloc[index]["atlas_name"]
        return {
            "image": image,
            "ap": ap_value, 
            "ox": ox,
            "oy": oy,
            "oz": oz,
            "ux": ux,
            "uy": uy,
            "uz": uz,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "atlas_name": atlas_name,
            "filename": filename,
        }

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self._labels)
