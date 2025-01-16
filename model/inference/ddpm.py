#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import ipdb  # noqa: F401
import torch
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Optional, Union
import inspect

rescale_fn = {
    "zero": lambda x: 0,
    "identity": lambda x: x,
    "square": lambda x: x**2,
    "square_root": lambda x: torch.sqrt(x),
}

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def shift_scale_denormalize(x, shift, scale):
    '''denormalize the normalized data using the shfit/mean and scale/std.'''
    return x / scale + shift


def inference_ddpm_call_varmod(
    model,
    scheduler,
    device,
    data=None,
    num_inference_steps=1000,
    guidance_scale=1.0,
    cfg=None,
):
    """
    Implements DDPM-style inference.

    To get multiple samples, batch the images multiple times.

    Args:
        model: Ray Diffuser.
        images (torch.Tensor): (B, N, C, H, W).
        crop_parameters (torch.Tensor): (B, N, 4) or None.
        pbar (bool): If True, shows a progress bar.
    """ 
    # batch_size, num_images, num_channel, num_patches_x, num_patches_y = data['rgb']['data'].shape
    batch_size, num_images = data['view_id'].shape[:2]
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    cond_mods = [mod for mod in model.modalities if mod in data['conds']]
    gen_mods = [mod for mod in model.modalities if mod in data['gens']]
    use_rgb, use_ray, use_depth = 'rgb' in gen_mods, 'ray' in gen_mods, 'depth' in gen_mods

    x_t = data
    with torch.no_grad():
        for t in tqdm(timesteps): 
            # predict the noise residual        
            mmod_preds = model(
                t=t.repeat(batch_size)-1,
                data={**x_t, 'uncond': False},
            )
            noise_pred_rgb = mmod_preds['rgb'] if use_rgb else None
            noise_pred_ray = mmod_preds['ray'] if use_ray else None
            noise_pred_depth = mmod_preds['depth'] if use_depth else None

            if guidance_scale > 1.0:
                mmod_preds_uncond = model(
                    t=t.repeat(batch_size)-1,
                    data={**x_t, 'uncond': True},
                )
                noise_pred_rgb_uncond = mmod_preds_uncond['rgb'] if use_rgb else None
                noise_pred_ray_uncond = mmod_preds_uncond['ray'] if use_ray else None
                noise_pred_depth_uncond = mmod_preds_uncond['depth'] if use_depth else None
                noise_pred_rgb = noise_pred_rgb_uncond + guidance_scale * (noise_pred_rgb - noise_pred_rgb_uncond) if use_rgb else None
                noise_pred_ray = noise_pred_ray_uncond + guidance_scale * (noise_pred_ray - noise_pred_ray_uncond) if use_ray else None
                noise_pred_depth = noise_pred_depth_uncond + guidance_scale * (noise_pred_depth - noise_pred_depth_uncond) if use_depth else None

            # compute the previous noisy sample x_t -> x_t-1
            if use_rgb:
                x_t_rgb = scheduler.step(noise_pred_rgb.flatten(0, 1).float(), t-1, x_t['gens']['rgb'].flatten(0, 1).float(), return_dict=False)[0].to(noise_pred_rgb.dtype)
                x_t_rgb = x_t_rgb.reshape((batch_size, num_images) + x_t_rgb.shape[1:]) 
                x_t['gens']['rgb'] = x_t_rgb
                x_t['gens']['rgb_mask'] = mmod_preds['rgb_mask']
            if use_ray:
                x_t_ray = scheduler.step(noise_pred_ray.flatten(0, 1).float(), t-1, x_t['gens']['ray'].flatten(0, 1).float(), return_dict=False)[0].to(noise_pred_ray.dtype)
                x_t_ray = x_t_ray.reshape((batch_size, num_images) + x_t_ray.shape[1:])
                x_t['gens']['ray'] = x_t_ray
                x_t['gens']['ray_mask'] = mmod_preds['ray_mask']
            if use_depth:
                x_t_depth = scheduler.step(noise_pred_depth.flatten(0, 1).float(), t-1, x_t['gens']['depth'][:, :, 0:1].flatten(0, 1).float(), return_dict=False)[0].to(noise_pred_depth.dtype)
                x_t_depth = x_t_depth.reshape((batch_size, num_images) + x_t_depth.shape[1:])
                x_t['gens']['depth'] = torch.cat([x_t_depth, x_t['gens']['depth'][:, :, 1:2]], dim=2)
                x_t['gens']['depth_mask'] = mmod_preds['depth_mask']   
        
        # shift-scale denormalize
        if use_rgb:
            x_t['gens']['rgb'] = shift_scale_denormalize(x_t['gens']['rgb'], cfg.data.shift_scales.rgb[0], cfg.data.shift_scales.rgb[1])   
        if use_ray:
            if cfg.data.use_plucker:
                x_t['gens']['ray'][:, :, :3] = shift_scale_denormalize(x_t['gens']['ray'][:, :, :3], cfg.data.shift_scales.ray.dirs[0], cfg.data.shift_scales.ray.dirs[1])
                x_t['gens']['ray'][:, :, 3:] = shift_scale_denormalize(x_t['gens']['ray'][:, :, 3:], cfg.data.shift_scales.ray.moms[0], cfg.data.shift_scales.ray.moms[1])
            else:
                x_t['gens']['ray'][:, :, :3] = shift_scale_denormalize(x_t['gens']['ray'][:, :, :3], cfg.data.shift_scales.ray.origins[0], cfg.data.shift_scales.ray.origins[1])
                x_t['gens']['ray'][:, :, 3:] = shift_scale_denormalize(x_t['gens']['ray'][:, :, 3:], cfg.data.shift_scales.ray.directions[0], cfg.data.shift_scales.ray.directions[1])
        if use_depth:
            x_t['gens']['depth'][:, :, :1] = shift_scale_denormalize(x_t['gens']['depth'][:, :, :1], cfg.data.shift_scales.depth[0], cfg.data.shift_scales.depth[1])       


    return x_t