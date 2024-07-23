from typing import Literal

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader

from data_source import DataSource, DataSourceType
from dataset_utils import load_atlas_reference, transform_atlas_volume
from fix_empty_image import FixEmptyImage
from pair_dataset import BrainwaysPairDataset
from single_dataset import BrainwaysSingleDataset
from utils import debugger_is_active


class BrainwaysDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_sources: dict[Literal["train", "val", "test"], list[DataSource]],
        bias_nearby_samples_p: float = 0.0,
        atlas_weights: dict[str, float] | None = None,
        source_type_weights: dict[DataSourceType, float] | None = None,
        num_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size

        # Load atlas
        atlases = self.load_atlases()

        # Set number of workers
        self.num_workers = self.set_num_workers(num_workers)

        # Define transformations
        self.train_transform = self.define_train_transform()
        self.eval_transform = self.define_eval_transform()

        # Prepare test atlas
        self.pred_atlases = self.prepare_pred_atlases(atlases)

        # Prepare datasets
        self.train_dataset = BrainwaysPairDataset(
            data_sources=data_sources["train"],
            transform=self.train_transform,
            bias_nearby_samples_p=bias_nearby_samples_p,
            atlas_weights=atlas_weights,
            source_type_weights=source_type_weights,
        )
        self.val_dataset = BrainwaysSingleDataset(
            data_sources=data_sources["val"],
            transform=self.eval_transform,
        )
        self.test_dataset = BrainwaysSingleDataset(
            data_sources=data_sources["test"],
            transform=self.eval_transform,
        )

    def load_atlases(self) -> dict[str, NDArray[np.float32]]:
        atlases = {
            atlas_name: load_atlas_reference(atlas_name)
            for atlas_name in ["whs_sd_rat_39um", "allen_mouse_25um"]
        }
        return atlases

    def set_num_workers(self, num_workers: int):
        return 0 if debugger_is_active() else num_workers

    @staticmethod
    def define_train_transform():
        return A.Compose(
            [
                A.Resize(224, 224, interpolation=cv2.INTER_AREA),
                A.HorizontalFlip(),
                A.Affine(
                    scale=(0.5, 1.0), rotate=(-15, 15), mode=cv2.BORDER_REFLECT_101
                ),
                A.Normalize(normalization="image"),
                FixEmptyImage(),
                A.Normalize(normalization="min_max"),
                A.FromFloat(dtype="uint8"),
                # A.HistogramMatching(
                #     reference_images=[self.reference_image],
                #     read_fn=lambda x: x,
                #     always_apply=True,
                # ),
                A.Equalize(always_apply=True),
                A.ElasticTransform(alpha=500, sigma=20, alpha_affine=0),
                A.ToFloat(),
                FixEmptyImage(),
                A.Normalize(normalization="min_max"),
                A.InvertImg(),
                A.ToRGB(),
                A.Normalize(max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def define_eval_transform():
        return A.Compose(
            [
                A.Resize(224, 224, interpolation=cv2.INTER_AREA),
                A.Normalize(normalization="image"),
                FixEmptyImage(),
                A.Normalize(normalization="min_max"),
                A.FromFloat(dtype="uint8"),
                # A.HistogramMatching(
                #     reference_images=[self.reference_image],
                #     read_fn=lambda x: x,
                #     always_apply=True,
                # ),
                A.Equalize(always_apply=True),
                A.ToFloat(),
                FixEmptyImage(),
                A.Normalize(normalization="min_max"),
                A.ToRGB(),
                A.Normalize(max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

    def prepare_pred_atlases(
        self, atlases: dict[str, NDArray[np.float32]]
    ) -> dict[str, torch.Tensor]:
        transformed_atlases: dict[str, torch.Tensor] = {}
        for atlas_name, atlas_reference in atlases.items():
            transformed_atlases[atlas_name] = transform_atlas_volume(
                atlas_volume=atlas_reference, transform=self.eval_transform
            )
        return transformed_atlases

    def train_dataloader(self):
        return self.prepare_dataloader(self.train_dataset, num_workers=self.num_workers)

    def val_dataloader(self):
        return self.prepare_dataloader(self.val_dataset, num_workers=2)

    def test_dataloader(self):
        return self.prepare_dataloader(self.test_dataset, num_workers=2)

    def prepare_dataloader(self, dataset, num_workers: int):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=False,
        )
