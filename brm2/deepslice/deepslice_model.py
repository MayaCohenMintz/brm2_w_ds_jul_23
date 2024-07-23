import torch
import torch.nn as nn

from brainways_backbone import BrainwaysBackbone
from custom_types import ModelOutput


class DeepSliceModel(nn.Module):
    def __init__(self, backbone: BrainwaysBackbone, inner_dim: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.backbone.feature_size, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 9), 
        )

    def forward(self, image: torch.Tensor) -> ModelOutput:
        # defines the computation performed at every call.
        # regarding channels: each channel represents a color. In RGB images each pixel has 3 channels,
        # grayscale is single channel. 
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): Input image A with dimensions (batch_size, channels, height, width).

        Returns:
            ModelOutput: Output of the model, including loss, logits, and predictions. Loss is a single value tensor, logits and predictions have dimensions (batch_size, num_labels).
        """
        # Q: check what the meaning of A's dimensions are, and what is batch_size here
        features = self.backbone.feature_extractor(image=image) 
        # Q: check what backbone is used here, it should be Xception. Figure out where this is instantiated in the code.
        # Also note that BrainwaysBackbone.feature_extractor is not yet implemented.
        logits = self.classifier(features)
        # preds = torch.sigmoid(logits)
        preds = logits

        return ModelOutput(logits=logits, preds=preds, features=features) # the output is a dictionary 
        # containing the three params as keys. 
