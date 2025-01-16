#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import io
import cv2
import base64
import plotly
import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
from model.load import load_model
from model.utils.rays import Rays, rays_to_cameras_homography
from utils.train_utils import model_inference
from utils.data_utils import DataHandler, tensor_recursive_to
from utils.vis import view_color_coded_images_from_tensor
from data import Preprocessor


HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
<body><img src="data:image/png;charset=utf-8;base64,{image_encoded}"/>
{plotly_html}</body></html>"""


def plotly_scene_visualization_dual(pred_camera, gt_camera, scale=0.03):
    num_frames = len(pred_camera)
    camera = {}
    R_pred, T_pred = pred_camera.R, pred_camera.T
    for i in range(num_frames):
        camera[i] = PerspectiveCameras(R=R_pred[i, None], T=T_pred[i, None])
    if gt_camera is not None:
        R_gt, T_gt = gt_camera.R, gt_camera.T
        for i in range(num_frames):
            camera[i + num_frames] = PerspectiveCameras(R=R_gt[i, None], T=T_gt[i, None])

    fig = plot_scene(
        {"scene": camera},
        camera_scale=scale,
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")
    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
    if gt_camera is not None:
        for i in range(num_frames):
            fig.data[i + num_frames].line.color = matplotlib.colors.to_hex((0.0, 0.0, 0.0, 1.0))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--exp_name", type=str, default='pose-estimation')
    parser.add_argument("--data_path", type=str, default=None, help="examples/co3dv2-samples/31_1359_4114")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--config", type=str, default="configs/config_stage3.yaml", help="Path to training config yaml file.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="inference cfg. 1.0 denotes not use cfg")
    parser.add_argument("--default_fov", type=float, default=60.0)
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
    preprocessor = Preprocessor(cfg, fov=args.default_fov)
        
    # Get data
    data = preprocessor(args.data_path, input_type='multi-view')
    data_handler = DataHandler(data)
    
    # Hyper-parameters setting & Mod flag editing
    # In this example, 8 views (id = 0, 1, 2, 3, 4, 5, 6, 7) are used for inference
    # Set mod flags as 'cggggggg,cccccccc,xxxxxxxx' denotes states of 'rgb,pose,depth',
    # where 'c' denotes condition, 'g' denotes generation, 'x' denotes not used
    # The order of letters in each modality denotes the state order
    # e.g., 'ccggx' denotes view 0-1 as condition, view 2-3 as generation, and view 4 as not used
    # would be auto-cutted based on the view numbers
    used_view_ids = torch.arange(8)
    mod_flags = 'cccccccc,cggggggg,xxxxxxxx'

    # Set random seed
    SEED = np.random.randint(0, 2147483647)
        
    # Inference
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, batch = \
        model_inference(models, data_handler, used_view_ids, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=args.guidance_scale, seed=SEED)

    # Write paired visualizations
    num_view = len(used_view_ids)
    gt_part = np_image[0][:512*num_view, :512*3]
    pred_part = np_image[0][-512*num_view:, 512:512*3]
    # from left to right: gt_rgb - gt_pose (dir + mom) - pred_pose (dir + mom)
    concat_images = np.concatenate([gt_part, pred_part], axis=1)
    file_name = f"{data_handler('scene_id')}-{SEED}-compare.png"
    cv2.imwrite(os.path.join(exp_folder, file_name), concat_images[..., ::-1])

    # Save camera visualization html following RayDiffuison
    gt_camera = data_handler('gt_pyt3d_camera')[0][used_view_ids] if data_handler('gt_pyt3d_camera') else None
    gt_rays = data_handler('cond_rays')[0][used_view_ids].float().cpu()
    pred_ray = mmod_preds['gens']['ray'][0].float().cpu()
    pred_ray[0] = gt_rays[0]
    # create camera from rays
    pred_camera = rays_to_cameras_homography(
        Rays.from_spatial(pred_ray),
        crop_parameters=None,
        num_patches_x=cfg.data.raymap_size,
        num_patches_y=cfg.data.raymap_size,
    )
    fig = plotly_scene_visualization_dual(pred_camera, gt_camera, scale=0.1)
    output_path = os.path.join(exp_folder, f"{data_handler('scene_id')}-{SEED}-cameras-vis.html")
    html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    s = io.BytesIO()
    images = torch.nn.functional.interpolate(data_handler('cond_image')[0][used_view_ids], size=(128, 128), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    view_color_coded_images_from_tensor(images)
    plt.savefig(s, format="png", bbox_inches="tight")
    plt.close()
    image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    with open(output_path, "w") as f:
        s = HTML_TEMPLATE.format(
            image_encoded=image_encoded,
            plotly_html=html_plot,
        )
        f.write(s)                
