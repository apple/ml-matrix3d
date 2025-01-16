#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import imageio
from omegaconf import OmegaConf
import trimesh
import open3d as o3d
from pytorch3d.renderer import PerspectiveCameras
from collections import Counter
from torchvision.transforms.functional import normalize

from data import Preprocessor
from model.load import load_model
from utils.train_utils import model_inference
from utils.data_utils import DataHandler, get_rgbd_point_cloud_numpy, save_compare_image
import utils.camera_utils as camera_utils
from IS_Net.models import *


def find_anchor_indices(dists, num_anchors=3):
    sorted_indices = np.argsort(dists, axis=-1)[:, :num_anchors]
    # must include the nearest anchors
    nearest_indices = np.unique(sorted_indices[:, 0])
    if len(nearest_indices) < num_anchors:
        # then we select other anchors
        rest_num_select = num_anchors - len(nearest_indices)
        flatten_sorted_indices = np.setdiff1d(sorted_indices.flatten(), nearest_indices, assume_unique=True)
        index_counts = Counter(flatten_sorted_indices)
        rest_most_common_indices = [idx for idx, _ in index_counts.most_common(rest_num_select)]
        nearest_indices = np.concatenate([nearest_indices, rest_most_common_indices])
    elif len(nearest_indices) > num_anchors:
        # then we need to remove some anchors
        flatten_sorted_indices = sorted_indices[:, 0]
        index_counts = Counter(flatten_sorted_indices)
        nearest_indices = [idx for idx, _ in index_counts.most_common(num_anchors)]
    return sorted(nearest_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--exp_name", type=str, default='single-to-3d')
    parser.add_argument("--data_path", type=str, default=None, help="examples/single-view/skull.png")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--config", type=str, default="configs/config_stage3.yaml", help="Path to training config yaml file.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--default_fov", type=float, default=60.0)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--nvs_with_depth_cond", type=int, default=1)
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

    # Load DIS model
    DISNet = ISNetDIS()
    DISNet.load_state_dict(torch.load("checkpoints/isnet-general-use.pth"))
    DISNet = DISNet.to(device)
    
    # Load data pre-processor
    preprocessor = Preprocessor(cfg, fov=args.default_fov)
        
    # Get data
    data = preprocessor(args.data_path, input_type='single-view')
    data_handler = DataHandler(data)

    # Set seed
    SEED = np.random.randint(0, 2147483647) if args.random_seed else 2024  # positive int32 range for random seed if applied
    
    ######## RUN! ########
    ############################################
    ### STEP 0: Generate Camera Trajectory  ####
    ############################################
    num_samples = args.num_samples
    num_cond = 1   # single-view input
    num_in, num_out = 3, 5   # 3in5out
    H, W, DINO_size = 512, 512, 896
    scene_images = [None for _ in range(num_samples)]
    # create folders
    scene_id = data_handler('scene_id')
    scene_folder = os.path.join(exp_folder, f'{scene_id}')
    scene_subfolder_image = os.path.join(scene_folder, 'images')
    scene_subfolder_depth = os.path.join(scene_folder, 'depths')
    scene_subfolder_mask = os.path.join(scene_folder, 'masks')
    os.makedirs(scene_folder, exist_ok=True)
    os.makedirs(scene_subfolder_image, exist_ok=True)
    os.makedirs(scene_subfolder_depth, exist_ok=True)
    os.makedirs(scene_subfolder_mask, exist_ok=True)
    # save seed
    np.savetxt(os.path.join(scene_folder, 'seed.txt'), np.array([SEED]), fmt='%d')
        
    # create obrit pytorch3d cameras
    angles = np.linspace(0, 360, num_samples, endpoint=False)
    elevations = np.zeros(num_samples)
    R, T = camera_utils.set_pytorch3d_cameras_eye_at_up(azimuths=angles, elevations=elevations) 
    pyt3d_camera = data_handler('gt_pyt3d_camera')[0]
    render_camera =  PerspectiveCameras(
        R=R,
        T=T,
        focal_length=pyt3d_camera.focal_length.repeat(num_samples, 1),
        principal_point=pyt3d_camera.principal_point.repeat(num_samples, 1),
        image_size=pyt3d_camera.image_size.repeat(num_samples, 1),
    )
    positions = render_camera.get_camera_center().cpu().numpy()   
    gen_rays = preprocessor.gen_raymap_from_camera(
        pyt3d_camera=render_camera, num_patches_x=cfg.modalities.ray.width, num_patches_y=cfg.modalities.ray.height, h=512, w=512)    
    gen_cam_indices = torch.arange(1, num_samples)
    # save to nerfstudio json file for later 3dgs reconstruction
    camera_utils.write_pyt3d_camera_to_nerfstudio_json(
        scene_folder, pyt3d_camera, render_camera[gen_cam_indices], has_ply=True, has_mask=True, has_depth=True)
    # save ref frames
    ref_imgs = F.interpolate(data_handler('cond_image')[0][:num_cond], (512, 512), mode="bilinear").permute(0, 2, 3, 1)
    ref_imgs = ((ref_imgs * 0.5 + 0.5) * 255.).cpu().numpy().astype(np.uint8)
    for j in range(num_cond):
        cv2.imwrite(os.path.join(scene_subfolder_image, f'ref_frame_{j:04d}.png'), ref_imgs[j][..., ::-1])
    # NOTE: take-place data, won't be used in the 3dgs reconstruction actually
    for j in range(num_samples-num_cond):
        np.save(os.path.join(scene_subfolder_depth, f'frame_{j:04d}.npy'), np.zeros([128, 128], np.float32))
    for j in range(num_cond):
        np.save(os.path.join(scene_subfolder_depth, f'ref_frame_{j:04d}.npy'), np.zeros([128, 128], np.float32))             
    
    # generate all required batch data at first, then select by index for each run
    # currently, only rays are required
    data_handler.pad_batch_data_using_first_value(num_samples)
    data_handler.update('cond_rays', torch.arange(num_cond, num_samples), gen_rays[num_cond:])
    data_handler.update('cond_image', torch.arange(num_cond, num_samples), -1)    
    
    ######################################################################
    ### Step 1: 1in7out -- first generate anchor views (RGB and depth) ###
    ######################################################################
    ### RGB part
    num_anchors = 7
    num_in, num_out = 8 - num_anchors, num_anchors
    anchor_views = np.linspace(0, num_samples, num_anchors + 1, endpoint=False).astype(np.int32)
    mod_flags = 'cggggggg,cccccccc,xxxxxxxx'
    print('selected anchor views:', anchor_views)
    
    # Inference & Save visualizations
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
        model_inference(models, data_handler, anchor_views, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=1.5, seed=SEED)
    save_compare_image(np_image, os.path.join(scene_folder, 'batch_generated_images', f'{scene_id}-anchor.png'))
            
    # Save anchor results for later generation
    with torch.no_grad():
        anchor_rgb_vae = mmod_preds['gens']['rgb'][:, num_in:]
        anchor_gen_rgb = models['vae'].decode(anchor_rgb_vae.flatten(0, 1) / models['vae'].config.scaling_factor, return_dict=False)[0]
        anchor_gen_rgb = F.interpolate(anchor_gen_rgb, size=(DINO_size, DINO_size), mode='bilinear').clamp(-1.0, 1.0).unflatten(0, (1, num_out))
        anchor_gen_rgb_dino = models['feature_extractor'](anchor_gen_rgb, autoresize=False) # (B, N, C, h, w)
        anchor_rgb_cond = torch.cat([anchor_gen_rgb_dino, anchor_rgb_vae], dim=2) 
        # add first reference
        anchor_rgb_cond = torch.cat([batch['data']['conds']['rgb'][:, 0:1], anchor_rgb_cond], dim=1)
    # overwrite the batch raw data using generated results
    data_handler.update('cond_image', anchor_views[num_in:], anchor_gen_rgb.float().cpu())

    # Save results
    all_anchor_gen_rgbs = [None for _ in range(num_anchors + num_cond)]
    for k in range(num_out):
        global_gen_id = anchor_views[num_in + k] - num_cond
        rgb_gen = pred_rgb[0, k+num_cond].float().cpu().numpy() * 255
        cv2.imwrite(os.path.join(scene_subfolder_image, f'frame_{global_gen_id:04d}.png'), rgb_gen[..., ::-1])
        scene_images[anchor_views[num_in + k]] = rgb_gen
        all_anchor_gen_rgbs[k+num_cond] = rgb_gen[None]
    scene_images[0] = np_image[0][:H, :H]
    all_anchor_gen_rgbs[0] = np_image[0][:H, :H][None]
    all_anchor_gen_rgbs = np.concatenate(all_anchor_gen_rgbs)

    ### Depth part
    mod_flags = 'cccccccc,cccccccc,gggggggg'
    
    # Inference & Save visualizations
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
        model_inference(models, data_handler, anchor_views, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=1.0, seed=SEED)
    save_compare_image(np_image, os.path.join(scene_folder, 'batch_generated_images', f'{scene_id}-anchor-depth.png'))
    
    # Save point cloud
    depth_cam = render_camera[torch.from_numpy(anchor_views).long()]
    pred_images = torch.from_numpy(all_anchor_gen_rgbs).permute(0, 3, 1, 2)
    # DIS mask
    DIS_images = F.interpolate(pred_images, (1024, 1024), mode='bilinear') / 255.
    DIS_images = normalize(DIS_images, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)
    with torch.no_grad():
        DIS_mask = DISNet(DIS_images)[0][0].cpu()
        DIS_mask_rgb = F.interpolate(DIS_mask, (H, W), mode='nearest')
        DIS_mask = F.interpolate(DIS_mask, (cfg.modalities.depth.height, cfg.modalities.depth.width), mode='nearest')
    pred_depths = 1.0 / mmod_preds['gens']['depth'][0].cpu()[:, 0:1]    
    # back-project & apply mask
    points, colors = get_rgbd_point_cloud_numpy(depth_cam, pred_images, pred_depths, depth_masks=DIS_mask, mask_thr=0.5)
    # remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.colors = o3d.utility.Vector3dVector(points), o3d.utility.Vector3dVector(colors)
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    inlier_points = np.asarray(cl.points).astype(np.float32)
    inlier_colors = np.asarray(cl.colors).astype(np.uint8)
    # write into a single ply file
    output_path = os.path.join(scene_folder, 'ref_pred_pointcloud.ply')
    combined_ply = trimesh.PointCloud(inlier_points, inlier_colors)
    _ = combined_ply.export(output_path)
    # write into masks folder
    masks = DIS_mask_rgb[:, 0].cpu().float().cpu().numpy() * 255
    for k in range(num_out):
        global_gen_id = anchor_views[num_in + k] - num_cond
        cv2.imwrite(os.path.join(scene_subfolder_mask, f'frame_{global_gen_id:04d}.png'), masks[k+1]) 
    cv2.imwrite(os.path.join(scene_subfolder_mask, f'ref_frame_{0:04d}.png'), masks[0]) 

    #####################################################################
    ### Step 2: then generate other views using selected anchor views ###
    #####################################################################
    num_in, num_out = 3, 5 # 3in5out
    mod_flags = 'cccggggg,cccccccc,xxxxxxxx'
    if args.nvs_with_depth_cond:
        all_gen_dis = mmod_preds['gens']['depth'][0][:, 0:1].float().cpu()
        data_handler.update('cond_depth', anchor_views, all_gen_dis)
        data_handler.update('gen_depth', anchor_views, all_gen_dis)
        mod_flags = 'cccggggg,cccccccc,cccccccc'
    rest_views = np.setdiff1d(np.arange(num_samples), anchor_views)
    num_runs = int(np.ceil(len(rest_views) / num_out))
    anchor_positions = positions[anchor_views][None]
    for idx in range(num_runs):
        idx_st, idx_ed = idx * num_out, min((idx + 1) * num_out, len(rest_views))
        num_gen = min(num_out, idx_ed - idx_st)
        gen_views = rest_views[idx_st : idx_ed]
        # select nearest 3 anchor views
        gen_positions = positions[gen_views]
        dists = np.linalg.norm(gen_positions[:, None, :] - anchor_positions, axis=-1)
        selected = find_anchor_indices(dists, num_anchors=num_in)
        view_indices = np.concatenate([anchor_views[selected], gen_views])
        print(f'run {idx}, current view indices: {view_indices}')
        
        # Inference & Save visualizations
        # NOTE: Here we hard-code the model_inference function by manually overwriting condition rgb latents 
        # with generated values to avoid applying vae encoder & decoder again.
        # This is a simple-fix and would not affect other tasks
        np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
            model_inference(models, data_handler, view_indices, mod_flags, preprocessor, cfg, args, 
                            device, weight_dtype, guidance_scale=1.5, seed=SEED, vae_encoding=anchor_rgb_cond[:, selected])
        save_compare_image(np_image, os.path.join(scene_folder, 'batch_generated_images', f'{scene_id}-run{idx}.png'))   
             
        # Save results
        pred_images = pred_rgb[0, num_in:num_in+num_gen].permute(0, 3, 1, 2).float()
        DIS_images = F.interpolate(pred_images, (1024, 1024), mode='bilinear')
        DIS_images = normalize(DIS_images, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)
        with torch.no_grad():
            DIS_mask = DISNet(DIS_images)[0][0].cpu()
            DIS_mask_rgb = F.interpolate(DIS_mask, (H, W), mode='nearest')
            masks = DIS_mask_rgb[:, 0].cpu().float().cpu().numpy() * 255
        for k in range(num_gen):
            scene_images[gen_views[k]] = rgb_gen
            global_gen_id = gen_views[k] - num_cond
            rgb_gen = pred_rgb[0, k+num_in].float().cpu().numpy() * 255
            cv2.imwrite(os.path.join(scene_subfolder_image, f'frame_{global_gen_id:04d}.png'), rgb_gen[..., ::-1])
            cv2.imwrite(os.path.join(scene_subfolder_mask, f'frame_{global_gen_id:04d}.png'), masks[k]) 
            
    # Save video
    render_file = os.path.join(scene_folder, f'diffusion-samples.mp4')
    imageio.mimsave(render_file, scene_images, fps=30)                       