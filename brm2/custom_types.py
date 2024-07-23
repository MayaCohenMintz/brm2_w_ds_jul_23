from typing import TypedDict

import torch


class ModelOutput(TypedDict):
    logits: torch.Tensor
    preds: torch.Tensor
    features: torch.Tensor

# the three params are the keys, and their values are of type torch.Tensor