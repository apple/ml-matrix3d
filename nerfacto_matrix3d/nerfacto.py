#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import depth_ranking_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

@dataclass
class NerfactoMatrix3DModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config with matrix3d integration"""

    _target: Type = field(default_factory=lambda: NerfactoMatrix3DModel)

    # Matrix3D specific parameters
    depth_l1_lambda: float = 0.0
    """Lambda for L1 depth loss"""
    depth_ranking_lambda: float = 0.0
    """Lambda for depth ranking loss"""
    l1_lambda_on_captured_views: float = 1.0
    """Lambda for L1 image loss on captured views"""
    l1_lambda_on_generation_views: float = 0.1
    """Lambda for L1 image loss on generation views"""
    output_depth_during_training: bool = False
    """Whether to output depth during training"""

class NerfactoMatrix3DModel(NerfactoModel):
    """Nerfacto model with matrix3d integration

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoMatrix3DModelConfig

    def __init__(
        self,
        config: NerfactoMatrix3DModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Overridden get_outputs method with matrix3d specific processing"""
        # Outputs from original NerfactoModel
        outputs = super().get_outputs(ray_bundle)

        # Add depth output during training if requested
        if self.config.output_depth_during_training and self.training:
            if hasattr(self, 'renderer_depth') and 'weights' in outputs and ray_bundle.samples is not None:
                outputs["depth"] = self.renderer_depth(outputs["weights"], ray_bundle.samples)

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Overridden loss dict with matrix3d specific losses"""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Matrix3D specific losses - simplified for compatibility
        if self.training and "camera_dist" in batch and "rgb_loss" in loss_dict:
            # Identify captured cameras (camera_dist == -1)
            captured_cam = batch["camera_dist"] == -1
            if torch.any(captured_cam):
                # Scale RGB loss based on whether it's a captured view or generated view
                rgb_loss = loss_dict["rgb_loss"]
                # Apply different weights and compute weighted mean
                loss_dict["rgb_loss"] = rgb_loss * self.config.l1_lambda_on_captured_views

        return loss_dict
