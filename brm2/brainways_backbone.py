from abc import ABC

import torch
from torch import nn


class BrainwaysBackbone(nn.Module, ABC):
    def feature_extractor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): An image tensor expected to have dimensions (batch_size, channels, height, width)

        Returns:
            torch.Tensor: A tensor with dimensions (batch_size, features)
        """
        raise NotImplementedError()

    @property
    def feature_size(self) -> int:
        """
        Returns the size of the feature.
        
        This method should be implemented by subclasses to return the size of the feature.
        
        Returns:
            int: Feature size.
        """
        raise NotImplementedError()
