from typing import Optional

import torch
import torch.nn as nn

from brainways_backbone import BrainwaysBackbone
from constants import AP_LIMITS
from custom_types import ModelOutput


class BrainwaysModel(nn.Module):
    def __init__(
        self, backbone: BrainwaysBackbone, pred_atlases: dict[str, torch.Tensor], inner_dim: int = 256
    ) -> None:
        super().__init__()
        self.backbone = backbone

        if self.backbone.feature_size > inner_dim:
            downsample_op = nn.Linear(self.backbone.feature_size, inner_dim)
        else:
            downsample_op = nn.Identity()

        self.feature_dim_reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            downsample_op,
        )

        self.classifier = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 1),
        )

        self._init_atlases(pred_atlases)

    def _init_atlases(self, pred_atlases: dict[str, torch.Tensor]) -> None:
        for atlas_name, atlas in pred_atlases.items():
            buffer_name = f"_atlas_{atlas_name}"
            self.register_buffer(buffer_name, atlas, persistent=False)

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> ModelOutput:
        """
        Forward pass of the model.

        Args:
            image_a (torch.Tensor): Input image A with dimensions (batch_size, channels, height, width).
            image_b (torch.Tensor): Input image B with dimensions (batch_size, channels, height, width).
            labels (Optional[torch.Tensor]): Ground truth labels with dimensions (batch_size)

        Returns:
            ModelOutput: Output of the model, including loss, logits, and predictions. Loss is a single value tensor, logits and predictions have dimensions (batch_size, num_labels).
        """
        embed_a = self.backbone.feature_extractor(image=image_a)
        embed_a_reduced = self.feature_dim_reduce(embed_a)
        embed_b = self.backbone.feature_extractor(image=image_b)
        embed_b_reduced = self.feature_dim_reduce(embed_b)
        linear_input = torch.cat([embed_a_reduced, embed_b_reduced], dim=1)
        logits = self.classifier(linear_input)
        preds = torch.tanh(logits)

        return ModelOutput(logits=logits, preds=preds, features=linear_input)

    def predict(
        self, image: torch.Tensor, atlas_name: str, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the output based on the given input image and atlas.

        Args:
            image (torch.Tensor): The input image with dimensions (batch_size, channels, height, width).
            atlas_name (str): The name of the atlas.
            return_features (bool): Whether to return the features or not.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]: The predicted output or a tuple of the predicted output and the features.

        """
        with torch.no_grad():
            min_ap, max_ap = AP_LIMITS[atlas_name]
            atlas = self.get_atlas(atlas_name)
            low = torch.full((len(image),), min_ap, device=image.device)
            high = torch.full((len(image),), max_ap, device=image.device)
            while (low <= high).any():
                mid = (low + high) // 2
                atlas_slice = atlas[mid]
                output = self.forward(image, atlas_slice)
                pred = output["preds"].squeeze(1)
                features = output["features"]
                high = torch.where((pred < 0) & (low <= high), mid - 1, high)
                low = torch.where((pred >= 0) & (low <= high), mid + 1, low)

        if return_features:
            return low, features
        else:
            return low

    def get_atlas(self, atlas_name: str):
        return getattr(self, f"_atlas_{atlas_name}")
