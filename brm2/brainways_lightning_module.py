import os
import csv
from typing import Any, Sequence

import lightning as L
from lightning.pytorch.callbacks import BackboneFinetuning
import torch
from torch import nn

from brainways_backbone import BrainwaysBackbone
from brainways_model import BrainwaysModel


class BrainwaysLightningModule(L.LightningModule):
    def __init__(
        self,
        backbone: BrainwaysBackbone,
        pred_atlases: dict[str, torch.Tensor],
        backbone_finetuning: BackboneFinetuning | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.loss_fn = nn.MSELoss()

        self.model = BrainwaysModel(backbone=self.backbone, pred_atlases=pred_atlases)
        self.atlas_names = list(pred_atlases.keys())
        self.backbone_finetuning = backbone_finetuning

    def configure_callbacks(self) -> Sequence[L.Callback] | L.Callback:
        if self.backbone_finetuning is not None:
            return self.backbone_finetuning
        else:
            return []

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        output = self.model(image_a=batch["image_a"], image_b=batch["image_b"])

        loss = self.loss_fn(output["preds"], batch["label"].float())
        self.log(
            f"train/loss", loss.mean(), prog_bar=True, batch_size=len(batch["image_a"])
        )
        for atlas_name in self.atlas_names:
            atlas_name_mask = [a == atlas_name for a in batch["atlas_name"]]
            loss_atlas = loss[atlas_name_mask].mean()
            self.log(
                f"train/{atlas_name}/loss", loss_atlas, batch_size=sum(atlas_name_mask)
            )
        return loss

    def _save_metrics_to_csv(
        self, batch: dict[str, Any], pred: torch.Tensor, metrics: dict, filename: str
    ):
        csv_file_path = os.path.join(self.logger.log_dir, filename)
        fieldnames = [
            "atlas_name",
            "filename",
            "prediction",
            "ground_truth",
            "l1",
        ]
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write headers only if file is created now (not appended)
            if not file_exists:
                writer.writeheader()
            for an, fn, p, g, l in zip(
                batch["atlas_name"],
                batch["filename"],
                pred.tolist(),
                batch["ap"].tolist(),
                metrics["l1"].tolist(),
            ):
                writer.writerow(
                    {
                        "atlas_name": an,
                        "filename": fn,
                        "prediction": p,
                        "ground_truth": g,
                        "l1": l,
                    }
                )

    def _log_val_test_metrics(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        prefix: str,
        save_metrics_to_csv: bool = False,
        batch: dict[str, Any] | None = None,
    ):
        l1 = (torch.abs(gt - pred)).float()
        metrics = {"l1": l1}
        if save_metrics_to_csv:
            assert batch is not None
            self._save_metrics_to_csv(
                batch=batch,
                pred=pred,
                metrics=metrics,
                filename=prefix.replace("/", "_") + ".csv",
            )
        tensorboard_metrics = {
            f"{prefix}/{key}": value.mean() for key, value in metrics.items()
        }
        self.log_dict(tensorboard_metrics, prog_bar=True, batch_size=len(gt))

    def _get_atlas_batch(self, batch, atlas_name):
        atlas_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                atlas_batch[key] = value[[a == atlas_name for a in batch["atlas_name"]]]
            else:
                atlas_batch[key] = [
                    value[i]
                    for i, a in enumerate(batch["atlas_name"])
                    if a == atlas_name
                ]
        return atlas_batch

    def _process_atlas_batch(self, atlas_batch, atlas_name: str, split: str):
        atlas_pred = self.model.predict(
            image=atlas_batch["image"], atlas_name=atlas_name
        )
        atlas_gt = atlas_batch["ap"].squeeze()
        self._log_val_test_metrics(
            gt=atlas_gt,
            pred=atlas_pred,
            prefix=f"{split}/{atlas_name}",
            save_metrics_to_csv=split == "test",
            batch=atlas_batch,
        )
        return atlas_gt, atlas_pred

    def _val_test_step(self, batch, batch_idx, split: str):
        all_gt, all_pred = [], []
        for atlas_name in self.atlas_names:
            atlas_batch = self._get_atlas_batch(batch, atlas_name)
            if len(atlas_batch["image"]) > 0:
                atlas_gt, atlas_pred = self._process_atlas_batch(
                    atlas_batch, atlas_name, split
                )
                all_gt.append(atlas_gt)
                all_pred.append(atlas_pred)

        self._log_val_test_metrics(
            gt=torch.cat(all_gt), pred=torch.cat(all_pred), prefix=f"{split}/all"
        )

    def validation_step(self, batch, batch_idx):
        self._val_test_step(batch=batch, batch_idx=batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self._val_test_step(batch=batch, batch_idx=batch_idx, split="test")
