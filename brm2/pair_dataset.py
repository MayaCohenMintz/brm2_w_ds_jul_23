import random
from typing import Callable, TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from constants import AP_LIMITS
from data_source import DataSource, DataSourceType
from dataset_utils import parse_data_sources, read_image


class PairDatasetSample(TypedDict):
    image_a: torch.Tensor
    image_b: torch.Tensor
    ap_a: float
    ap_b: float
    label: float
    atlas_name: str


class BrainwaysPairDataset(IterableDataset):
    def __init__(
        self,
        data_sources: list[DataSource],
        transform: Callable,
        bias_nearby_samples_p: float = 0.0,
        atlas_weights: dict[str, float] | None = None,
        source_type_weights: dict[DataSourceType, float] | None = None,
    ):
        self._atlas_names = list(set([source["atlas_name"] for source in data_sources]))
        if atlas_weights is not None:
            if set(atlas_weights.keys()) != set(self._atlas_names):
                raise ValueError(
                    "Atlas weights must be provided for all atlases. "
                    f"Expected {self._atlas_names}, got {set(atlas_weights.keys())}."
                )

        self._labels = parse_data_sources(data_sources)
        self._grouped_labels: dict[tuple[str, DataSourceType], pd.DataFrame] = {}
        for atlas_name in self._atlas_names:
            for type in ["real", "synth"]:
                self._grouped_labels[(atlas_name, type)] = self._labels[
                    (self._labels["atlas_name"] == atlas_name)
                    & (self._labels["type"] == type)
                ]
        self._transform = transform
        self._bias_nearby_samples_p = bias_nearby_samples_p
        self._atlas_weights = atlas_weights
        self._source_type_weights = source_type_weights

    def _generate_sample_pair(self) -> PairDatasetSample:
        if random.random() < self._bias_nearby_samples_p:
            image_a, ap_a, atlas_name = self._generate_random_sample()
            image_b, ap_b, atlas_name = self._generate_biased_random_sample(
                near_ap=ap_a, atlas_name=atlas_name, source_type="synth"
            )
        else:
            image_a, ap_a, atlas_name = self._generate_random_sample()
            image_b, ap_b, atlas_name = self._generate_random_sample(
                atlas_name=atlas_name, source_type="synth"
            )
        label = self._get_pair_label(ap_a=ap_a, ap_b=ap_b)

        return {
            "image_a": image_a,
            "image_b": image_b,
            "atlas_name": atlas_name,
            "label": label,
            "ap_a": ap_a,
            "ap_b": ap_b,
        }

    def _generate_random_sample(
        self, atlas_name: str | None = None, source_type: DataSourceType | None = None
    ) -> tuple[torch.Tensor, float, str]:
        if source_type is None:
            source_type = self._generate_random_source_type()
        if atlas_name is None:
            atlas_name = self._generate_random_atlas_name()
        
        index = np.random.choice(self._grouped_labels[(atlas_name, source_type)].index)
        return self._generate_sample(index)

    def _generate_biased_random_sample(
        self, near_ap: float, atlas_name: str, source_type: DataSourceType | None = None
    ) -> tuple[torch.Tensor, float, str]:
        """Generate a random sample with a bias towards the given AP."""
        if source_type is None:
            source_type = self._generate_random_source_type()

        ap = np.random.normal(loc=near_ap, scale=20)
        min_ap, max_ap = AP_LIMITS[atlas_name]
        ap = int(np.clip(ap, min_ap, max_ap - 1))

        # Calculate the absolute differences
        differences = np.abs(self._grouped_labels[(atlas_name, source_type)]["ap"] - ap)

        # Get the indices of all samples whose value has a difference of up to 1
        up_to_one_indices = np.where(differences <= 1)[0]

        if len(up_to_one_indices) > 0:
            # If such samples exist, randomly choose an index from these indices
            index = np.random.choice(up_to_one_indices)
        else:
            # If no such samples exist, choose the nearest neighbor
            sorted_indices = differences.argsort()
            index = sorted_indices[0]

        return self._generate_sample(index)

    def _generate_random_source_type(self):
        all_source_types = ["real", "synth"]
        if self._source_type_weights is None:
            return random.choice(all_source_types)
        else:
            weights = [self._source_type_weights[type] for type in all_source_types]
            return random.choices(all_source_types, weights=weights, k=1)[0]

    def _generate_random_atlas_name(self):
        if self._atlas_weights is None:
            return random.choice(self._atlas_names)
        else:
            weights = [self._atlas_weights[name] for name in self._atlas_names]
            return random.choices(self._atlas_names, weights=weights, k=1)[0]

    def _get_pair_label(self, ap_a: float, ap_b: float) -> float:
        return float(np.clip(ap_a - ap_b, -50, 50) / 50)

    def _generate_sample(self, index: int) -> tuple[torch.Tensor, float, str]:
        atlas_name = str(self._labels.loc[index]["atlas_name"])
        image = read_image(self._labels.iloc[index].filename)
        image = self._transform(image=image)["image"]
        ap = float(self._labels.loc[index]["ap"])
        return image, ap, atlas_name

    def _iterator(self):
        while True:
            try:
                yield self._generate_sample_pair()
            except Exception as e:
                print(f"Error generating sample: {e}")

    def __iter__(self):
        return iter(self._iterator())
