#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import cv2
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from model.load import load_model
from utils.train_utils import model_inference
from utils.data_utils import DataHandler
from data import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--exp_name", type=str, default='novel-view-synthesis')
    parser.add_argument("--data_path", type=str, default=None, help="examples/co3dv2-samples/31_1359_4114")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--config", type=str, default="configs/config_stage3.yaml", help="Path to training config yaml file.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="inference cfg. 1.0 denotes not use cfg")
    args = parser.parse_args()

    # Make experiment directory
    exp_folder = f'./results/{args.exp_name}'
    os.makedirs(exp_folder, exist_ok=True)

    # Set cuda and mixed precision
    device = torch.device(f"cuda:{args.gpu}")
    if args.mixed_precision == "no":
        weight_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load config and model
    cfg = OmegaConf.load(args.config)
    models = load_model(cfg, args.checkpoint_path, device=device, weight_dtype=weight_dtype)
    
    # Load data pre-processor
    preprocessor = Preprocessor(cfg)
        
    # Get data
    data = preprocessor(args.data_path, input_type='multi-view')
    data_handler = DataHandler(data)
    
    # Hyper-parameters setting & Mod flag editing
    # In this example, 4 views (id = 0, 1, 2, 3) are used for inference
    # Set mod flags as 'cggggggg,cccccccc,xxxxxxxx' denotes states of 'rgb,pose,depth',
    # where 'c' denotes condition, 'g' denotes generation, 'x' denotes not used
    # The order of letters in each modality denotes the state order
    # e.g., 'ccggx' denotes view 0-1 as condition, view 2-3 as generation, and view 4 as not used
    # would be auto-cutted based on the view numbers
    used_view_ids = torch.arange(4)
    mod_flags = 'cggggggg,cccccccc,cccccccc'

    # Set random seed
    SEED = np.random.randint(0, 2147483647)
        
    # Inference
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
        model_inference(models, data_handler, used_view_ids, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=args.guidance_scale, seed=SEED)

    # Write paired visualizations
    num_view = len(used_view_ids)
    gt_part = np_image[0][:512*num_view, :512*3]
    pred_part = np_image[0][-512*num_view:, :512]
    # from left to right: gt_rgb - gt_pose (dir + mom) - pred_rgb
    concat_images = np.concatenate([gt_part, pred_part], axis=1)
    file_name = f"{data_handler('scene_id')}-{SEED}-compare.png"
    cv2.imwrite(os.path.join(exp_folder, file_name), concat_images[..., ::-1])