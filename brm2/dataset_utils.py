import os
import sys
from pathlib import Path

import albumentations as A
import albucore.utils
import cv2
import numpy as np
import pandas as pd
import torch
from bg_atlasapi import BrainGlobeAtlas
from numpy.typing import NDArray

from constants import AP_LIMITS
from data_source import DataSource


def parse_data_sources(data_sources: list[DataSource]) -> pd.DataFrame:
    """
    Parses the given list of data sources and returns a concatenated pandas DataFrame.

    Args:
        data_sources (list[DataSource]): A list of DataSource objects representing the data sources.

    Returns:
        pd.DataFrame: A concatenated pandas DataFrame containing the parsed data.

    """
    labels = []
    for data_source in data_sources: # from yml file
        labels.append(load_and_filter_labels(data_source))
    return pd.concat(labels).reset_index(drop=True) 


def load_and_filter_labels(data_source: DataSource) -> pd.DataFrame: 
    """
    Load and filter the labels from the given path.

    Args:
        data_source (DataSource): The data source containing the paths and other information.

    Returns:
        pd.DataFrame: The filtered labels dataframe.

    """
    labels = pd.read_csv(data_source["labels_path"]) 
    print(f"labels columns: {labels.columns}")
    print(f"labels: {labels}")

    # MAYA - added this: Check if 'filename' column exists, if not, check 'Filenames' and rename if found
    if "filename" not in labels.columns:
        if "Filenames" in labels.columns:
            labels.rename(columns={"Filenames": "filename"}, inplace=True)
        else:
            raise ValueError("Neither 'filename' nor 'Filenames' columns found in the data source")

    if "channel" not in labels.columns:
        labels["channel"] = "?"

    # Add full path to the filename
    labels["filename"] = labels["filename"].apply(
        lambda x: os.path.join(data_source["images_path"], x)
    )

    # Add atlas name and data type
    labels["atlas_name"] = data_source["atlas_name"]
    labels["type"] = data_source["type"]

    print(f"labels column labels: {labels.columns}")
    if "ox" not in labels.columns:
        sys.exit("ox was not a column in the labels file. Exiting.")

    # CHANGE or remove the following two sections on filtering by AP - not needed since we are not looking at AP 

    # Filter out labels based on AP limits
    ap_limits = AP_LIMITS[data_source["atlas_name"]]
    ap_limited_labels = labels[
        (labels["ap"] >= ap_limits[0]) & (labels["ap"] < ap_limits[1])
    ]

    # Filter out invalid entries
    if data_source["type"] == "real":
        valid_labels = ap_limited_labels[ap_limited_labels["valid"].isin(["yes", True])]
    else:
        valid_labels = ap_limited_labels

    filtered_labels = valid_labels[valid_labels["channel"] != "Fluorgold tracer"]
    # print(f"filtered column labels: {filtered_labels.columns}")

    return filtered_labels


def read_image(image_path: Path | str) -> np.ndarray: 
    # this overrides a lightning method. 
    # Reads image into a 3 dimensional RGB or grayscale Tensor. Optionally converts the image to the 
    # desired format. The values of the output tensor are uint8 in [0, 255]
    """
    Read an image from the given file path and return it as a NumPy array.

    Args:
        image_path (Path or str): The path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array.

    Raises:
        ValueError: If the image is empty (all pixels are black).

    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    # if image.max() == 0:
    #     raise ValueError(f"Image {image_path} is empty.")
    if albucore.utils.is_rgb_image(image):
        image = (
            np.dot(image, [0.2989, 0.5870, 0.1140])
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
    return image


def create_atlas_slice(
    ap: int, atlas: torch.Tensor, transform: A.TransformType | None = None
) -> torch.Tensor:
    # extracts a specific slice from the atlas at a given AP index and applies optional transformations.
    # CHANGE - I think this needs to receive O,U,V vectors, should translate them to ap axis, and return corresponding ap slice.
    """
    Create a slice of the atlas image at a given AP (Anterior-Posterior) index.

    Args:
        ap (int): The AP index of the slice to create.
        atlas (torch.Tensor): The atlas image. ### MAYA - I think this means atlas.reference
        transform (A.TransformType | None, optional): An optional transformation to apply to the slice. Defaults to None.

    Returns:
        torch.Tensor: The created atlas slice.

    Raises:
        ValueError: If the atlas image at the given AP index is empty.
    """

    atlas_slice = atlas[ap] # from GPT: this slice is a tensor of shape (y,x) where y is height and x is width.
    # CHANGE - need to get slice by O,U,V vecs (do I?)
    # also - what is this slice? A numpy array/ tensor containing what? 
    if atlas_slice.max() == 0:
        raise ValueError(f"Atlas image at AP {ap} is empty")
    if transform is not None:
        atlas_slice = transform(image=atlas_slice)["image"]
    return atlas_slice


def load_atlas_reference(atlas_name: str) -> NDArray[np.float32]:
    """
    Load the atlas reference volume for a given atlas name.

    Parameters:
        atlas_name (str): The name of the atlas.

    Returns:
        NDArray[np.float32]: The loaded atlas reference as a NumPy array of type np.float32 with shape (z, y, x).
    """
    # from GPT: z = number of slices (anterior-posterior axis), y = height of slice (inferior-superior axis)
    # x = width of slice (left-right axis)
    bg_atlas = BrainGlobeAtlas(atlas_name)
    atlas_reference = (bg_atlas.reference / bg_atlas.reference.max()).astype(np.float32)
    return atlas_reference


def transform_atlas_volume(
    atlas_volume: NDArray[np.float32], transform: A.TransformType
) -> torch.Tensor:
    transformed_atlas_volume = torch.stack(
        [transform(image=atlas_slice)["image"] for atlas_slice in atlas_volume]
    )
    return transformed_atlas_volume
