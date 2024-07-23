import argparse

from pytorch_lightning import Trainer

from brainways_lightning_module import BrainwaysLightningModule
from brainways_datamodule import BrainwaysDataModule


def main(checkpoint_path):
    datamodule = BrainwaysDataModule(batch_size=32)
    lit_module = BrainwaysLightningModule.load_from_checkpoint(
        checkpoint_path, test_atlas=datamodule.test_atlas
    )
    trainer = Trainer(limit_test_batches=500)
    test_results = trainer.test(model=lit_module, datamodule=datamodule)
    print(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file", required=True
    )
    args = parser.parse_args()

    main(args.checkpoint)
