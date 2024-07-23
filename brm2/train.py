import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from brainways_lightning_module import BrainwaysLightningModule
from brainways_datamodule import BrainwaysDataModule


def main():
    torch.set_float32_matmul_precision("medium")

    monitor = "v_loss"

    datamodule = BrainwaysDataModule(batch_size=32)
    lit_module = BrainwaysLightningModule(test_atlas=datamodule.test_atlas)
    callbacks = [
        ModelCheckpoint(monitor=monitor),
        EarlyStopping(monitor=monitor, patience=7),
    ]

    trainer = pl.Trainer(
        limit_val_batches=300,
        val_check_interval=500,
        precision="16-mixed",
        callbacks=callbacks,
    )
    # tuner = pl.tuner.Tuner(trainer)
    # tuner.scale_batch_size(lit_module, datamodule=datamodule)

    trainer.fit(model=lit_module, datamodule=datamodule)
    trainer.test(model=lit_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
