import itertools
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def process_csv(csv_file: Path):
    df = pd.read_csv(csv_file)

    # Filter out rows that were excluded from the DeepSlice analysis
    if csv_file.name == "PCP2.csv":
        df = df[~df["Filenames"].str.contains("_s173|_s176", regex=True)]
    elif csv_file.name == "GLTa.csv":
        df = df[~df["Filenames"].str.contains("s107|s112|s117", regex=True)]

    o = df[["ox", "oy", "oz"]].values
    u = df[["ux", "uy", "uz"]].values
    v = df[["vx", "vy", "vz"]].values

    center = np.array([0.5, 0.5, 1])
    matrix = np.stack([u, v, o], axis=1)
    atlas_center_coords = center @ matrix
    df["ap"] = 527 - atlas_center_coords[:, 1]
    df = df[(df["ap"] >= 0) & (df["ap"] <= 527)]
    processed_df = df[["Filenames", "ap"]].rename(columns={"Filenames": "filename"})
    return processed_df


def write_output_csv(
    df: pd.DataFrame, output_root: Path, data_root: Path, output_labels_filename: str
):
    df["valid"] = "yes"
    df["channel"] = "?"

    output_csv_index = []
    for image_path in df.index:
        output_image_dir = output_root / "images" / Path(image_path).parent
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_image_filename = Path(image_path).stem + ".jpg"
        output_image_path = output_image_dir / output_image_filename
        output_csv_index.append(output_image_filename)
        if output_image_path.exists():
            continue
        resized_image = cv2.imread(str(data_root / image_path))
        resized_image = cv2.resize(
            resized_image, (512, 512), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(str(output_image_path), resized_image)

    df.index = output_csv_index
    df.index.name = "filename"
    df.to_csv(output_root / output_labels_filename)


def main():
    data_root = Path("/home/ben/python/brm2/data/deepslice_raw/")
    output_root = Path("/home/ben/python/brm2/data/deepslice_processed/")
    operators_root = data_root / "Operator_Alignments"

    processed_csvs = defaultdict(list)
    for operator_dir in operators_root.iterdir():
        for csv_file in operator_dir.glob("*.csv"):
            processed_csvs[csv_file.stem].append(process_csv(csv_file))

    operator_averaged_csvs = {}
    for stain, dfs in processed_csvs.items():
        operator_averaged_csvs[stain] = pd.concat(dfs).groupby(by="filename").mean()

    val_stains = ["GLTa", "PCP2", "CAMKII"]
    test_stains = ["DAB", "ISH", "Myelin", "PITX3"]

    validation_csvs = [operator_averaged_csvs[s] for s in val_stains]
    test_csvs = [operator_averaged_csvs[s] for s in test_stains]

    print(f"Validation: {sum(len(operator_averaged_csvs[i]) for i in val_stains)}")
    print(f"Test: {sum(len(operator_averaged_csvs[i]) for i in test_stains)}")

    write_output_csv(
        df=pd.concat(validation_csvs),
        output_root=output_root,
        data_root=data_root,
        output_labels_filename="val_labels.csv",
    )
    write_output_csv(
        df=pd.concat(test_csvs),
        output_root=output_root,
        data_root=data_root,
        output_labels_filename="test_labels.csv",
    )


if __name__ == "__main__":
    main()
