import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_dir(dir_path: Path) -> pd.DataFrame | None:
    root = Path("/data/cfos/allen-resized/")
    labels_path = root / dir_path / "labels.csv"
    labels = None
    if labels_path.exists():
        try:
            labels = pd.read_csv(labels_path)
        except Exception as e:
            print(f"Error reading {labels_path}: {e}")
    return labels


def main():
    with open("/home/ben/python/allen-downloader/dataset_ids.txt") as f:
        dataset_ids = [line.strip() for line in f]

    # Create a pool of processes
    with mp.Pool(mp.cpu_count()) as pool:
        # Map the function to the directories
        all_labels = list(tqdm(pool.imap(process_dir, dataset_ids)))
        all_labels = [labels for labels in all_labels if labels is not None]
    all_labels_combined = pd.concat(all_labels)
    
    train_labels, val_test_labels = train_test_split(
        all_labels, test_size=0.02, random_state=42
    )
    val_labels, test_labels = train_test_split(
        val_test_labels, test_size=0.5, random_state=42
    )

    # Log the number of dataframes and records in each split
    print("Number of datasets:", len(all_labels))
    print("Number of images:", len(all_labels_combined))
    logger.info(f"Number of dataframes in train split: {len(train_labels)}")
    logger.info(
        f"Number of records in train split: {sum(len(df) for df in train_labels)}"
    )
    logger.info(f"Number of dataframes in validation split: {len(val_labels)}")
    logger.info(
        f"Number of records in validation split: {sum(len(df) for df in val_labels)}"
    )
    logger.info(f"Number of dataframes in test split: {len(test_labels)}")
    logger.info(
        f"Number of records in test split: {sum(len(df) for df in test_labels)}"
    )

    # Find duplicates
    duplicates = pd.concat(all_labels).duplicated()
    num_duplicates = duplicates.sum()
    logger.info(f"Number of duplicates: {num_duplicates}")

    # Create directories
    out_dir = Path("/home/ben/python/brm2/data/allen/")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_labels_combined.to_csv(out_dir / "all_labels.csv", index=False)
    pd.concat(train_labels).to_csv(out_dir / "train_labels.csv", index=False)
    pd.concat(val_labels).to_csv(out_dir / "val_labels.csv", index=False)
    pd.concat(test_labels).to_csv(out_dir / "test_labels.csv", index=False)


if __name__ == "__main__":
    main()
