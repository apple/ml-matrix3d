#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

import torch
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.utils.rich_utils import CONSOLE

# Define a generic type variable for dataset
TDataset = TypeVar('TDataset', bound=DepthDataset)


@dataclass
class BatchFullImageDatamanagerConfig(VanillaDataManagerConfig):
    """Full images data manager config

    Args:
        num_image_each_iteration: number of images to sample each iteration
        max_num_iterations: maximum number of iterations for training
        cache_images_type: type of caching for images, either 'uint8' or 'float32'
    """

    num_image_each_iteration: int = 3
    """Number of images to sample each iteration"""
    max_num_iterations: int = 30000
    """Maximum number of iterations for training"""
    cache_images_type: Literal["float32", "uint8"] = "uint8"
    """Type of caching for images"""


class FullImageDatamanager(DataManager, Generic[TDataset]):
    """Full images data manager implementation

    Args:
        config: the DataManagerConfig used to instantiate class
        device: the device to load data to (cpu, cuda, mps, etc.)
        test_mode: bool that indicates whether the data manager is in test mode
        world_size: the number of processes
        local_rank: the rank of the current process
    """

    config: BatchFullImageDatamanagerConfig

    def __init__(
        self,
        config: BatchFullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.test_mode = test_mode
        self.world_size = world_size
        self.local_rank = local_rank

        super().__init__()

    def setup_train(self):
        """Sets up training dataloader"""
        pass

    def setup_eval(self):
        """Sets up evaluation dataloader"""
        pass

    def next_train(self, step: int) -> Tuple[Cameras, Dict[str, torch.Tensor]]:
        """Returns the next training batch

        Args:
            step: the training step
        """
        # Simplified implementation - would be replaced with actual implementation
        # similar to the one in splatfacto_matrix3d when used
        camera = Cameras(device=self.device)
        batch = {
            "image": torch.zeros((1, 1, 3), device=self.device),
            "camera_dist": torch.tensor([-1.0], device=self.device),
            "max_num_iterations": torch.tensor([self.config.max_num_iterations], device=self.device)
        }
        return camera, batch

    def next_eval(self, step: int) -> Tuple[Cameras, Dict[str, torch.Tensor]]:
        """Returns the next evaluation batch

        Args:
            step: the evaluation step
        """
        return self.next_train(step)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get parameter groups for the datamanager"""
        return {}
