from itertools import chain

import torch
from lightning.pytorch.cli import LightningCLI

from backbones import *  # noqa: F403
from brainways_datamodule import BrainwaysDataModule
from brainways_lightning_module import BrainwaysLightningModule


class BrainwaysLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.pred_atlases", "model.pred_atlases", apply_on="instantiate"
        )
        parser.add_argument("--monitor", type=str, help="Quantity to be monitored.")
        parser.add_argument(
            "--monitor_mode", type=str, help="Mode for monitoring quantity (min/max)."
        )

    @staticmethod
    def configure_optimizers(
        lightning_module: BrainwaysLightningModule, optimizer, lr_scheduler=None
    ):
        """Override to customize the :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lightning_module.backbone_finetuning is not None:
            assert len(optimizer.param_groups) == 1, f"Optimizer should have only one param group, got {len(optimizer.param_groups)}."
            optimizer.param_groups[0]["params"] = chain(
                lightning_module.model.feature_dim_reduce.parameters(),
                lightning_module.model.classifier.parameters(),
            )
        return LightningCLI.configure_optimizers(lightning_module, optimizer, lr_scheduler)


def main():
    torch.set_float32_matmul_precision("medium")
    BrainwaysLightningCLI(
        BrainwaysLightningModule,
        BrainwaysDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        # auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
