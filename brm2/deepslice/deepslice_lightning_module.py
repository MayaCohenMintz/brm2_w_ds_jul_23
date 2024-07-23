import os
import csv
from typing import Any, Sequence

import lightning as L
from lightning.pytorch.callbacks import BackboneFinetuning
import torch
from torch import nn

from brainways_backbone import BrainwaysBackbone
from deepslice.deepslice_model import DeepSliceModel


class DeepSliceLightningModule(L.LightningModule):
    def __init__(
        self,
        backbone: BrainwaysBackbone,
        backbone_finetuning: BackboneFinetuning | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = DeepSliceModel(backbone=self.backbone) # Q: how can the DS model work with self.backbone which
        # is of type BrainwaysBackbone? maybe it is taken from the yml?
        self.loss_fn = nn.MSELoss(reduction="none")
        self.backbone_finetuning = backbone_finetuning

    def configure_callbacks(self) -> Sequence[L.Callback] | L.Callback: # returns a callback or a series of callbacks
        if self.backbone_finetuning is not None:
            return self.backbone_finetuning # Q: why are the callbacks not the ones defined in the yml?
        else:
            return []

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # this function returns loss for the provided batch. 
        # batch is taken from train_dataloader in DataModule. Typically contains a dictionary with data and labels.  
        output = self.model(image=batch["image"]) # outputing a ModelOutput dict with logits, preds and features as keys. 
        # General note: when using the instance of the model like a function, it implicitly invokes the forward method.

        assert output["preds"].shape == batch["label"].shape, (
            f"Output shape {output['preds'].shape} does not match label shape "
            f"{batch['label'].shape}"
        )
        # ensures there is a model prediction for every gt label in the batch 

        unreduced_loss = self.loss_fn(output["preds"], batch["label"].float()) 
        # loss_fn creates a criterion that measures the MSE between each element in the input output["preds"] and target batch["label"].
        # there are batch_size elements in each of these. 
        # "unreduced" means it is not reduced to a single value, but rather returns a loss value for each element. 
        # to summarize: unreduced_loss tensor will have the shape (batch_size, num_labels), which is same
        # shape as output["preds"] and batch["labels"]. Each element in this tensor represents the MSE loss for the corresponding element in the predictions and ground truth labels.
        loss = unreduced_loss.mean() # here it is reduced to a single scalar value representing the average loss for the entire batch.  
        self.log(
            f"train/loss", loss, prog_bar=True, batch_size=len(batch["image"])
        )
        atlas_names = sorted(set(batch["atlas_name"]))
        for atlas_name in atlas_names:
            atlas_name_mask = [a == atlas_name for a in batch["atlas_name"]]
            loss_atlas = unreduced_loss[atlas_name_mask]
            self.log(
                f"train/{atlas_name}/loss", loss_atlas.mean(), batch_size=sum(atlas_name_mask)
            )
        return loss

    def _save_metrics_to_csv(
        self, batch: dict[str, Any], pred: torch.Tensor, metrics: dict, filename: str
    ):
        # what is the metrics dict? what is inside it?
        csv_file_path = os.path.join(self.logger.log_dir, filename)
        fieldnames = [
            "atlas_name",
            "filename",
            "prediction",
            "ground_truth",
            "l1", # Q: maybe change this to "metric" since we use distance for validation?
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
                batch["label"].tolist(),
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
        # calculates and logs the precision metric of the model prediction vs gt.
        l1 = (torch.abs(gt - pred)).float() # CHANGE this to our distance metric 
        metrics = {"l1": l1}
        if save_metrics_to_csv:
            assert batch is not None
            self._save_metrics_to_csv(
                batch=batch,
                pred=pred,
                metrics=metrics,
                filename=prefix.replace("/", "_") + ".csv",
            )
        tensorboard_metrics = { # CHANGE this as well maybe?
            f"{prefix}/{key}": value.mean() for key, value in metrics.items()
        }
        self.log_dict(tensorboard_metrics, prog_bar=True, batch_size=len(gt))

    def _get_atlas_batch(self, batch, atlas_name):
        # prepares subset of batch data by selecting elements where the corresponding atlas_name matches the specified atlas_name.
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
        # returns gt and model predictions of the batch
        atlas_pred = self.model(image=atlas_batch["image"])["preds"].squeeze()
        # atlas_gt = atlas_batch["label"].squeeze() # problem: here there are 9 labels (and not a single ap label), 
        # which are not called "label" but rather by vector names. Seems that model outputs 9 numbers as it should. 
        # I need to figure out: what type, shape etc should atlas_gt be?
        # MAYA - making prediction a stacking of the 9 numbers: 
        atlas_gt = torch.stack([atlas_batch["ox"], atlas_batch["oy"], atlas_batch["oz"], atlas_batch["ux"], atlas_batch["uy"], atlas_batch["uz"], atlas_batch["vx"], atlas_batch["vy"], atlas_batch["vz"]], dim=1).squeeze()

        self._log_val_test_metrics(
            gt=atlas_gt,
            pred=atlas_pred,
            prefix=f"{split}/{atlas_name}",
            save_metrics_to_csv=split == "test", # save the metrics if split == test
            batch=atlas_batch,
        )
        return atlas_gt, atlas_pred

    def _val_test_step(self, batch, batch_idx, split: str):
        all_gt, all_pred = [], []
        atlas_names = sorted(set(batch["atlas_name"]))
        for atlas_name in atlas_names:
            atlas_batch = self._get_atlas_batch(batch, atlas_name)
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
