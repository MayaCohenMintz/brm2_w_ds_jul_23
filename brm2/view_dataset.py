import argparse

import napari
import numpy as np

from brainways_lightning_module import BrainwaysLightningModule
from brainways_datamodule import BrainwaysDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    args = parser.parse_args()

    model = None
    if args.model:
        model = BrainwaysLightningModule.load_from_checkpoint(args.model).model

    datamodule = BrainwaysDataModule(batch_size=1)

    ds_iter = iter(datamodule.val_dataloader())
    samples = [next(ds_iter) for _ in range(50)]

    viewer = napari.Viewer()

    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 16
    viewer.text_overlay.color = (1.0, 1.0, 1.0, 1.0)
    viewer.text_overlay.position = "top_center"

    viewer.add_image(np.stack([s["image_a"] for s in samples]), name="image_a")
    viewer.add_image(
        np.stack([s["image_b"] for s in samples]), name="image_b", translate=(0, 0, 512)
    )
    viewer.reset_view()

    def on_index_changed(event):
        index = event.source.current_step[0]
        sample = samples[index]
        gt = float(sample["label"])
        if model is not None:
            output = model(
                sample["image_a"].cuda(),
                sample["image_b"].cuda(),
                sample["label"].cuda(),
            )
            pred = float(output["preds"])
            viewer.text_overlay.text = f"{gt=:.3f}, {pred=:.3f}"
        else:
            viewer.text_overlay.text = f"{gt=}"

    viewer.dims.events.current_step.connect(on_index_changed)
    napari.run()


if __name__ == "__main__":
    main()
