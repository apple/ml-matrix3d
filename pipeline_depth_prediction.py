#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import cv2
import argparse
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from model.load import load_model
from utils.train_utils import model_inference
from utils.data_utils import DataHandler, get_rgbd_point_cloud_numpy
from data import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--exp_name", type=str, default='depth-prediction')
    parser.add_argument("--data_path", type=str, default=None, help="examples/co3dv2-samples/31_1359_4114")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--config", type=str, default="configs/config_stage3.yaml", help="Path to training config yaml file.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="inference cfg. 1.0 denotes not use cfg")
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
    # In this example, 3 views (id = 0, 1, 2) are used for inference
    # Set mod flags as 'cggggggg,cccccccc,xxxxxxxx' denotes states of 'rgb,pose,depth',
    # where 'c' denotes condition, 'g' denotes generation, 'x' denotes not used
    # The order of letters in each modality denotes the state order
    # e.g., 'ccggx' denotes view 0-1 as condition, view 2-3 as generation, and view 4 as not used
    # would be auto-cutted based on the view numbers
    used_view_ids = torch.arange(3)
    mod_flags = 'cccccccc,cccccccc,gggggggg'
    
    # Set random seed
    SEED = np.random.randint(0, 2147483647)
        
    # Inference
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
        model_inference(models, data_handler, used_view_ids, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=args.guidance_scale, seed=SEED)

    # Write paired visualizations
    num_view = len(used_view_ids)
    gt_part = np_image[0][:512*num_view, :512*4]
    pred_part = np_image[0][-512*num_view:, 512*3:512*4]
    # from left to right: gt_rgb - gt_pose (dir + mom) - gt-depth - pred depth
    concat_images = np.concatenate([gt_part, pred_part], axis=1)
    file_name = f"{data_handler('scene_id')}-{SEED}-compare.png"
    cv2.imwrite(os.path.join(exp_folder, file_name), concat_images[..., ::-1])
    
    # Back-project depth images to point clouds
    camera = data_handler('gt_pyt3d_camera')[0][used_view_ids]
    mask = depth_mask[0].cpu()
    gt_images = data_handler('gen_image')[0][used_view_ids][mask] * 0.5 + 0.5
    # write predictions
    pred_depths = 1.0 / mmod_preds['gens']['depth'][0].cpu()[:, 0:1]
    pred_points, pred_colors = get_rgbd_point_cloud_numpy(camera, gt_images, pred_depths)
    output_path = os.path.join(exp_folder, f"{data_handler('scene_id')}-{SEED}-depth-pred.ply")
    combined_ply = trimesh.PointCloud(pred_points, pred_colors * 255)
    _ = combined_ply.export(output_path) 
    # write groundtruths
    gt_depths = 1.0 / data_handler('gen_depth')[0][used_view_ids][mask].cpu()
    gt_depth_masks = torch.logical_and(gt_depths > 0, ~torch.isinf(gt_depths)).to(gt_depths)
    gt_points, gt_colors = get_rgbd_point_cloud_numpy(camera, gt_images, gt_depths, depth_masks=gt_depth_masks, mask_thr=0.5)
    output_path = os.path.join(exp_folder, f"{data_handler('scene_id')}-depth-gt.ply")
    combined_ply = trimesh.PointCloud(gt_points, gt_colors * 255)
    _ = combined_ply.export(output_path)                