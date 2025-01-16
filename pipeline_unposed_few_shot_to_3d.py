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

from data import Preprocessor
from model.utils.rays import Rays, rays_to_cameras_homography
from utils.train_utils import model_inference
from model.load import load_model
from utils.data_utils import DataHandler, get_rgbd_point_cloud_numpy, save_compare_image
import utils.camera_utils as camera_utils



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--exp_name", type=str, default='unposed-few-shot-to-3d')
    parser.add_argument("--data_path", type=str, default=None, help="examples/unposed-samples/31_1359_4114")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--config", type=str, default="configs/config_stage3.yaml", help="Path to training config yaml file.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--nvs_with_depth_cond", type=int, default=1)
    parser.add_argument("--num_cond_images", type=int, default=3)
    parser.add_argument("--spline_scales", type=int, default=3)
    parser.add_argument("--num_depth_runs_for_init_depth", type=int, default=21)
    parser.add_argument("--use_loop_traj", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='co3dv2')
    
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

    # Set seed
    SEED = np.random.randint(0, 2147483647) if args.random_seed else 2024  # positive int32 range for random seed if applied
    
    ######## RUN! ########
    #################################################
    ### STEP 0: Generate pose for captured views ####
    #################################################
    num_samples = args.num_samples
    spline_scales = args.spline_scales
    num_cond = args.num_cond_images
    num_in = num_cond
    num_out = 8 - num_in   # max supported view number = 8
    H, W = 512, 512
    scene_images = [None for _ in range(num_samples)]
    # create folders
    scene_id = data_handler('scene_id')
    scene_folder = os.path.join(exp_folder, f'{scene_id}')
    scene_subfolder_image = os.path.join(scene_folder, 'images')
    scene_subfolder_depth = os.path.join(scene_folder, 'depths')
    os.makedirs(scene_folder, exist_ok=True)
    os.makedirs(scene_subfolder_image, exist_ok=True)
    os.makedirs(scene_subfolder_depth, exist_ok=True)
    # save seed
    np.savetxt(os.path.join(scene_folder, 'seed.txt'), np.array([SEED]), fmt='%d')

    # NOTE: you need to always set the first-view pose as condition!
    mod_flags = 'cccccccc,cggggggg,xxxxxxxx'
    train_ids = torch.arange(num_cond)

    # Inference & Save visualizations
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
        model_inference(models, data_handler, train_ids, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=1.5, seed=SEED)
    save_compare_image(np_image, os.path.join(scene_folder, 'batch_generated_images', f'{scene_id}-pose-ref.png'))

    # Create camera from pred rays
    gt_rays = data_handler('cond_rays')[0][train_ids].float().cpu()
    pred_ray = mmod_preds['gens']['ray'][0][train_ids].float().cpu()
    pred_ray[0] = gt_rays[0]
    pred_cams = rays_to_cameras_homography(
        Rays.from_spatial(pred_ray),
        crop_parameters=None,
        num_patches_x=cfg.data.raymap_size,
        num_patches_y=cfg.data.raymap_size,
    )
    pred_cams.image_size = torch.Tensor([H, W])[None].repeat(num_cond, 1)
    # rebuild camera rays
    pred_rays = preprocessor.gen_raymap_from_camera(
        pyt3d_camera=pred_cams, num_patches_x=cfg.modalities.ray.width, num_patches_y=cfg.modalities.ray.height, h=H, w=W)
               
    # Few-shot reconstruction: generate camera trajs
    scene_images = []
    if args.dataset == 'co3dv2' or args.dataset == 'arkitscenes':
        spline_camera = camera_utils.fit_spline_given_pyt3d_cameras(
            pred_cams, n_frames=num_samples, scales=spline_scales, tension=0.0, continuity=0.0, 
            bias=0.0, is_loop=args.use_loop_traj)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}. Please design a specific camera trajetory or apply an existing method for this dataset.')              
    gen_rays = preprocessor.gen_raymap_from_camera(
        pyt3d_camera=spline_camera, num_patches_x=cfg.modalities.ray.width, num_patches_y=cfg.modalities.ray.height, h=512, w=512)
    # save to nerfstudio json file for later reconstruction
    camera_utils.write_pyt3d_camera_to_nerfstudio_json(
        scene_folder, pred_cams, spline_camera, None, has_ply=True, has_mask=False, has_depth=True)
    ref_imgs = F.interpolate(data_handler('gen_image')[0][train_ids], (H, W), mode="bilinear").permute(0, 2, 3, 1)
    ref_imgs = ((ref_imgs * 0.5 + 0.5) * 255.).cpu().numpy().astype(np.uint8)
    for j in range(num_cond):
        cv2.imwrite(os.path.join(scene_subfolder_image, f'ref_frame_{j:04d}.png'), ref_imgs[j][..., ::-1])
    # NOTE: take-place data, won't be used in the 3dgs reconstruction
    for j in range(num_samples * spline_scales):
        np.save(os.path.join(scene_subfolder_depth, f'frame_{j:04d}.npy'), np.zeros([128, 128], np.float32))
    for j in range(num_cond):
        np.save(os.path.join(scene_subfolder_depth, f'ref_frame_{j:04d}.npy'), np.zeros([128, 128], np.float32)) 
                    
    # generate all required batch data at first, then select by index for each run
    total_views = num_cond + num_samples * spline_scales
    data_handler.pad_batch_data_using_first_value(total_views)
    # update pred cameras to data buffer
    data_handler.update('gen_rays', torch.arange(num_cond), pred_rays)
    data_handler.update('cond_rays', torch.arange(num_cond), pred_rays)
    data_handler.update('cond_rays', torch.arange(num_cond, total_views), gen_rays)
    data_handler.update('cond_image', torch.arange(num_cond, total_views), -1)   

    #################################################
    ### STEP 1: Generate Depth for captured views ###
    #################################################
    # generate ref frames depth first
    all_depths = []
    depth_cam_ref = pred_cams  # use pred cameras for all following tasks
    mod_flags = 'cccccccc,cccccccc,gggggggg'
    view_indices = torch.arange(num_cond)
    # multi-runs to alleviate sampling randomness
    for idx in range(args.num_depth_runs_for_init_depth):
        print('run iterations for reference view depth generation:', idx, ', gen view_indices:', view_indices)
        
        # Inference & Save visualizations
        np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
            model_inference(models, data_handler, train_ids, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=1.0, seed=SEED)
        save_compare_image(np_image, os.path.join(scene_folder, 'batch_generated_images', f'{scene_id}-depth-ref-run{idx}.png'))
        
        # Save results
        gt_images = data_handler.data['gen_image'][0][view_indices].cpu() * 0.5 + 0.5
        pred_depths = 1.0 / mmod_preds['gens']['depth'][0][view_indices].cpu()[:, 0:1]
        all_depths.append(pred_depths)                   

    # aggregate and median selection & save results
    final_depth = torch.stack(all_depths).median(dim=0, keepdim=True)[0][0]
    for j in range(num_cond): 
        np.save(os.path.join(scene_subfolder_depth, f'ref_frame_{j:04d}.npy'), final_depth[j, 0].float().cpu().numpy()) 
    pred_points_ref, pred_colors_gt = get_rgbd_point_cloud_numpy(depth_cam_ref, gt_images, final_depth)    
    combined_ply = trimesh.PointCloud(pred_points_ref, pred_colors_gt)
    output_path = os.path.join(scene_folder, f'ref_pred_pointcloud.ply')
    _ = combined_ply.export(output_path)

    # update generated depth to data buffer
    all_gen_dis = 1.0 / final_depth.float()
    data_handler.update('cond_depth', torch.arange(num_cond), all_gen_dis)
    data_handler.update('gen_depth', torch.arange(num_cond), all_gen_dis)

    ##############################################
    ### STEP 2: Generate RGB for splined views ###    
    ##############################################       
    num_out = 8 - num_cond
    if args.nvs_with_depth_cond:
        mod_flags = 'c' * num_cond + 'g' * num_out + ',cccccccc,' + 'c' * num_cond + 'x' * num_out
    else:
        mod_flags = 'c' * num_cond + 'g' * num_out + ',cccccccc,xxxxxxxx'
    
    num_runs = int(np.ceil(num_samples * spline_scales / num_out))
    rest_views = np.arange(total_views)
    for idx in range(num_runs):
        idx_st, idx_ed = num_cond + idx * num_out, num_cond + min((idx + 1) * num_out, num_samples * spline_scales)
        num_gen = min(num_out, idx_ed - idx_st)
        gen_views = rest_views[idx_st : idx_ed]
        view_indices = np.concatenate([np.arange(num_in), gen_views])   # init
        print('run iteration for rgb generation:', idx, ', gen view_indices:', view_indices) 

        # Inference & Save visualizations
        np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
            model_inference(models, data_handler, view_indices, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=1.5, seed=SEED)
        save_compare_image(np_image, os.path.join(scene_folder, 'batch_generated_images', f'{scene_id}-run{idx}.png'))

        # Save results
        for k in range(num_gen):
            global_gen_id = idx_st + k - num_cond
            rgb_gen = pred_rgb[0, k+num_in].float().cpu().numpy() * 255
            cv2.imwrite(os.path.join(scene_subfolder_image, f'frame_{global_gen_id:04d}.png'), rgb_gen[..., ::-1])
            scene_images.append(rgb_gen)
    
    # Save video
    render_file = os.path.join(scene_folder, f'diffusion-samples.mp4')
    imageio.mimsave(render_file, scene_images, fps=30) 