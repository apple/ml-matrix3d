#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import _shape_t
from transformers.activations import get_activation

# https://github.com/yenchenlin/dinov2-adaLN/commit/b195e7b7ebeefc0b249173b23734b1fb64227a9f
class AdaLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, mod_shape: _shape_t = None, eps: float = 0.00001, mod_act='gelu', elementwise_affine: bool = True, device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        if mod_shape is None:
            mod_shape = normalized_shape
        self.mod_linear = nn.Sequential(
            get_activation(mod_act),
            nn.Linear(mod_shape, 2 * normalized_shape, bias=True)
        ) if mod_shape > 0 else None

    def modulate(self, x, scale, shift):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def mod_init(self):
        nn.init.zeros_(self.mod_linear[-1].weight)  # TODO why cannot use .zeros_() here
        nn.init.zeros_(self.mod_linear[-1].bias)

    def forward(self, input: Tensor, modulation: Tensor = None) -> Tensor:
        normed = super().forward(input)
        if modulation is None or self.mod_linear is None:
            return normed
        scale, shift = self.mod_linear(modulation).chunk(2, dim=1)
        return self.modulate(normed, scale, shift)
