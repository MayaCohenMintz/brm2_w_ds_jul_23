## MAYA ADDITIONS for it to work on my laptop
# Ensure the current working directory is 'parent'
import os
import sys
os.chdir('/home/ben/python/maya/brm2_with_deepslice_cloned-main/brm2')
sys.path.append(os.path.abspath('.'))
## END OF MAYA ADDITIONS

import torch
from lightning.pytorch.cli import LightningCLI

from backbones import * 
from deepslice.deepslice_datamodule import DeepSliceDataModule
from deepslice.deepslice_lightning_module import DeepSliceLightningModule


class BrainwaysLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--monitor", type=str, help="Quantity to be monitored.") # Q: is this val/all/l1 ?
        parser.add_argument(
            "--monitor_mode", type=str, help="Mode for monitoring quantity (min/max)."
        )

    @staticmethod
    def configure_optimizers(
        lightning_module: DeepSliceLightningModule, optimizer, lr_scheduler=None # lr scheduler controls changes  in learning rate (the eta I learned about)
    ):
        """Override to customize the :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lightning_module.backbone_finetuning is not None:
            assert (
                len(optimizer.param_groups) == 1
            ), f"Optimizer should have only one param group, got {len(optimizer.param_groups)}."
            optimizer.param_groups[0][
                "params"
            ] = list(lightning_module.model.classifier.parameters())
        return LightningCLI.configure_optimizers(
            lightning_module, optimizer, lr_scheduler
        )


def main():
    torch.set_float32_matmul_precision("medium")
    BrainwaysLightningCLI(
        DeepSliceLightningModule,
        DeepSliceDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        # MAYA CHANGE (for my windows laptop): changing parser_mode from omegaconf to yaml.
        # auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
