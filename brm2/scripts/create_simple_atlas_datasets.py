from pathlib import Path

import cv2
import pandas as pd
from dataset_utils import load_atlas_reference


def create_atlas_dataset(atlas_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_atlas_reference(atlas_name)
    labels_df = []
    for ap in range(reference.shape[0]):
        atlas_slice = reference[ap]
        filename = f"{atlas_name}_{ap}.tif"
        atlas_slice_path = output_dir / filename
        cv2.imwrite(str(atlas_slice_path), atlas_slice)
        labels_df.append({"filename": filename, "ap": ap, "valid": "true", "channel": "NA"})
    labels_df = pd.DataFrame(labels_df)
    labels_df.to_csv(output_dir / "labels.csv", index=False)


for atlas_name in ["whs_sd_rat_39um", "allen_mouse_25um"]:
    create_atlas_dataset(atlas_name=atlas_name, output_dir=Path(f"data/atlases/{atlas_name}"))