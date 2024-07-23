from typing import Literal

import albumentations as A
import cv2
import lightning as L
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from data_source import DataSource
from deepslice.deepslice_dataset import DeepSliceDataset
from fix_empty_image import FixEmptyImage
from utils import debugger_is_active


class DeepSliceDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_sources: dict[Literal["train", "val", "test"], list[DataSource]],
        num_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size

        # Set number of workers
        self.num_workers = self.set_num_workers(num_workers)

        # Define transformations
        self.train_transform = self.define_train_transform()
        # defines transformations used to preprocess or augment data before feeding it into model during training
        self.eval_transform = self.define_eval_transform()
        # defines transformations for evaluation and testing data. They might differ from those used during training, 
        # often focusing on ensuring the data is appropriately prepared for model evaluation.

        # Prepare datasets
        self.train_dataset = DeepSliceDataset(
            data_sources=data_sources["train"],
            transform=self.train_transform,
        )
        self.val_dataset = DeepSliceDataset(
            data_sources=data_sources["val"],
            transform=self.eval_transform,
        )
        self.test_dataset = DeepSliceDataset(
            data_sources=data_sources["test"],
            transform=self.eval_transform,
        )

    def set_num_workers(self, num_workers: int):
        return 0 if debugger_is_active() else num_workers

    @staticmethod
    def define_train_transform():
        return A.Compose( # receives image that is represented as an array of pixels
            [
                A.Resize(299, 299, interpolation=cv2.INTER_AREA),
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
                A.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def define_eval_transform():
        return A.Compose(
            [
                A.Resize(299, 299, interpolation=cv2.INTER_AREA),
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
                A.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

    def train_dataloader(self):
        return self.prepare_dataloader(self.train_dataset, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return self.prepare_dataloader(self.val_dataset, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return self.prepare_dataloader(self.test_dataset, shuffle=False, num_workers=2)

    def prepare_dataloader(self, dataset, shuffle: bool, num_workers: int):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=False,
        )
