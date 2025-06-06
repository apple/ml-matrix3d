from __future__ import annotations

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig, CameraOptimizerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .nerfacto import NerfactoMatrix3DModelConfig

nerfacto_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto_matrix3d",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfactoMatrix3DModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfacto model with matrix3d integration for 3D reconstruction from sparse views",
)

# Keep backward compatibility
nerfacto_matrix3d = nerfacto_method