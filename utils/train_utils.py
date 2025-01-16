#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.utils import set_seed
from typing import List, Optional

from model.inference.ddpm import inference_ddpm_call_varmod
from utils.data_utils import tensor_recursive_to

def convert_array_to_pil(depth_map):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    mask = depth_map!=0
    # disp_map = 1/depth_map
    disp_map = depth_map
    vmax = np.percentile(disp_map[mask], 99)
    vmin = np.percentile(disp_map[mask], 1)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im / 255.


def batch_convert_array_to_pil(depth_maps):
    if type(depth_maps) == torch.Tensor:
        depth_maps_np = depth_maps.float().detach().cpu().numpy()
    elif type(depth_maps) == np.ndarray:
        depth_maps_np = copy.deepcopy(depth_maps)
    depth_out = []
    for depth_map in depth_maps_np:
        depth_out.append(convert_array_to_pil(depth_map))
    depth_out = np.stack(depth_out)
    
    if type(depth_maps) == torch.Tensor:
        depth_out = torch.from_numpy(depth_out).to(depth_maps.device).to(depth_maps.dtype)
    
    return depth_out


def vis_rays(dirs=None, moms=None, to_0_1=False):
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    moms = torch.nn.functional.normalize(moms, dim=-1)
    if to_0_1:
        return (dirs + 1) / 2, (moms + 1) / 2
    else:
        return dirs, moms


# modified from HunyuanDiTPipeline
def hunyuan_encode_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    device: torch.device = None,
    dtype: torch.dtype = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    max_sequence_length: Optional[int] = None,
    text_encoder_index: int = 0,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        dtype (`torch.dtype`):
            torch dtype
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        prompt_attention_mask (`torch.Tensor`, *optional*):
            Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
        negative_prompt_attention_mask (`torch.Tensor`, *optional*):
            Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
        max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
        text_encoder_index (`int`, *optional*):
            Index of the text encoder to use. `0` for clip and `1` for T5.
    """
    if max_sequence_length is None:
        if text_encoder_index == 0:
            max_length = 77
        if text_encoder_index == 1:
            max_length = 256
    else:
        max_length = max_sequence_length

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        #     text_input_ids, untruncated_ids
        # ):
        #     removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        #     logger.warning(
        #         "The following part of your input was truncated because CLIP can only handle sequences up to"
        #         f" {tokenizer.model_max_length} tokens: {removed_text}"
        #     )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=prompt_attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=negative_prompt_attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask


def shift_scale_normalize(x, shift, scale):
    '''normalize the data using the shfit/mean and scale/std.'''
    return (x - shift) * scale


# def shift_scale_denormalize(x, shift, scale):
#     '''denormalize the normalized data using the shfit/mean and scale/std.'''
#     return x / scale + shift


def flatten_string_list(nested_list):
    flat_list = []
    positions = []
    for i, sublist in enumerate(nested_list):
        for j, item in enumerate(sublist):
            if item != '':
                flat_list.append(item)
                positions.append((i, j))
    return flat_list, positions    


def flatten_image_data(image_data, mask):
    valid_indices = torch.nonzero(mask, as_tuple=True)
    valid_image_data = image_data[valid_indices]
    return valid_image_data, valid_indices


def add_noise(latents, args, noise_scheduler, timesteps=None):
    if type(latents) == list:
        to_list = True
        bs = len(latents)
        bs_sizes = [latent.shape[0] for latent in latents]
        latents = torch.cat(latents, dim=0)
        raise NotImplementedError("Not implemented for latents list")
    else:
        to_list = False
        # raise NotImplementedError("Not implemented for non-list latents")
        bs, bs_sizes = latents.shape[:2]
        latents = latents.reshape((bs * bs_sizes, *latents.shape[2:]))
        
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    if args.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += args.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )
    if args.input_perturbation:
        noise = noise + args.input_perturbation * torch.randn_like(noise)
    
    # Sample a random timestep for each prompt with nview image
    if timesteps is None:
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device)
    timesteps_repeat = timesteps.repeat_interleave(torch.tensor(bs_sizes, device=latents.device), dim=0)
    timesteps_repeat = timesteps_repeat.long()
    
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_repeat)
    
    # Prepare target
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "sample":
        target = latents
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps_repeat)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    
    noisy_latents = noisy_latents.reshape((bs, bs_sizes) + noisy_latents.shape[1:])
    target = target.reshape((bs, bs_sizes) + target.shape[1:])
    
    return noisy_latents, target, timesteps


def generate_gaussian_blob(size, sigma):
    S = size
    x = torch.arange(0, S, 1).float()
    y = torch.arange(0, S, 1).float()
    y = y[:, None]
    x0 = y0 = S // 2

    gaussian_blob = 1 - torch.exp(- ((x - x0)**2 + (y - y0)**2) / (2 * sigma**2 * S**2))

    return gaussian_blob


def prepare_train_data(batch, gen_mods, cond_mods, noise_scheduler, feature_extractor, vae, weight_dtype, cfg, args, is_val=False, blob_init=False,
                       tokenizer=None, text_encoder=None, tokenizer_2=None, text_encoder_2=None, random_view_id=False, device=None):
    # NOTE: convert local captions into right format
    # this is a **bug** of pytorch dataloader: https://stackoverflow.com/questions/64883998/
    # simple fix: no need to write a new collect_fn
    # batch['local_caption'] = [list(row) for row in zip(*batch['local_caption'])]

    data = {
            'gens': {},
            'conds': {},
            'mods_flags': batch['mods_flags'],
            'view_id': batch['view_id'],
            'uncond': batch.get('uncond', False),
        }
    bs, num_view = batch['view_id'].shape
    if device == None:
        if 'intrinsic' in batch:
            device = batch['intrinsic'].device
        elif 'gen_image' in batch:
            device = batch['gen_image'].device
    
    if random_view_id:
        random_view_ids = []
        for _ in range(bs):
            ids = list(range(1, cfg.model.pe_config.view.max))
            random.shuffle(ids)
            random_view_ids.append([0] + sorted(ids[:num_view-1]))
        random_view_ids = torch.tensor(random_view_ids, dtype=data['view_id'].dtype, device=data['view_id'].device)
        data['view_id'] = random_view_ids

    timesteps = None if is_val else torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,)).to(device)
    target = None if is_val else {}
    for mod in gen_mods:
        if mod == 'rgb':
            bs, nvg, nc, h, w = batch['gen_image'].shape
            # vae
            if 'vae' in batch:
                gen_rgb = batch['vae']
            else:
                valid_gen_image, valid_indices = flatten_image_data(batch['gen_image'].to(weight_dtype), batch['mods_flags']['rgb'] == 1)
                if len(valid_indices[0]) > 0:
                    with torch.no_grad():
                        valid_gen_rgb = vae.encode(valid_gen_image.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    gen_rgb = torch.zeros((bs, nvg) + valid_gen_rgb.shape[1:], dtype=weight_dtype, device=device)
                    gen_rgb[valid_indices] = valid_gen_rgb
                else:
                    size_down = batch['gen_image'].shape[-1] // 8
                    gen_rgb = torch.zeros(bs, nvg, 4, size_down, size_down, dtype=weight_dtype, device=device)
            gen_rgb = shift_scale_normalize(gen_rgb, cfg.data.shift_scales.rgb[0], cfg.data.shift_scales.rgb[1])
            rgb_valid = torch.ones_like(gen_rgb).bool()
            # add noise
            if is_val:
                if blob_init:
                    blob = generate_gaussian_blob(h, sigma=0.15).to(weight_dtype).to(device) * 2 - 1.
                    blob = blob.unsqueeze(0).unsqueeze(0).repeat(bs * nvg, 3, 1, 1)
                    with torch.no_grad():
                        gen_rgb = vae.encode(blob).latent_dist.sample() * vae.config.scaling_factor
                    gen_rgb = gen_rgb.reshape((bs, nvg) + gen_rgb.shape[1:])
                    gen_rgb = shift_scale_normalize(gen_rgb, cfg.data.shift_scales.rgb[0], cfg.data.shift_scales.rgb[1])
                    blob_time = torch.Tensor([999] * bs).long().to(device)
                    noisy_gen_rgb, target_rgb, _ = add_noise(gen_rgb, args, noise_scheduler, timesteps=blob_time)
                else:
                    noisy_gen_rgb = torch.randn_like(gen_rgb)
            else:
                noisy_gen_rgb, target_rgb, _ = add_noise(gen_rgb, args, noise_scheduler, timesteps=timesteps)
                target.update({'rgb': target_rgb, 'rgb_valid': rgb_valid})
            data['gens'].update({'rgb': noisy_gen_rgb, 'rgb_valid': rgb_valid})
        elif mod == 'ray':
            gen_ray = batch['gen_rays'].clone()
            if cfg.data.use_plucker:
                gen_ray[:, :, :3] = shift_scale_normalize(gen_ray[:, :, :3], cfg.data.shift_scales.ray.dirs[0], cfg.data.shift_scales.ray.dirs[1])
                gen_ray[:, :, 3:] = shift_scale_normalize(gen_ray[:, :, 3:], cfg.data.shift_scales.ray.moms[0], cfg.data.shift_scales.ray.moms[1])
            else:
                gen_ray[:, :, :3] = shift_scale_normalize(gen_ray[:, :, :3], cfg.data.shift_scales.ray.origins[0], cfg.data.shift_scales.ray.origins[1])
                gen_ray[:, :, 3:] = shift_scale_normalize(gen_ray[:, :, 3:], cfg.data.shift_scales.ray.directions[0], cfg.data.shift_scales.ray.directions[1])
            ray_valid = torch.ones_like(gen_ray).bool()
            # add noise
            if is_val:
                noisy_gen_ray= torch.randn_like(gen_ray)
            else:
                noisy_gen_ray, target_ray, _ = add_noise(gen_ray, args, noise_scheduler, timesteps=timesteps)
                target.update({'ray': target_ray, 'ray_valid': ray_valid})
            data['gens'].update({'ray': noisy_gen_ray, 'ray_valid': ray_valid})
        elif mod == 'depth':
            gen_depth = batch['gen_depth'].clone()
            depth_valid = (gen_depth > 0) & (gen_depth < 10)
            gen_depth = shift_scale_normalize(gen_depth, cfg.data.shift_scales.depth[0], cfg.data.shift_scales.depth[1])
            # add noise
            if is_val:
                noisy_gen_depth = torch.randn_like(gen_depth)
                depth_valid = torch.ones_like(gen_depth).float().to(gen_depth.device)
            else:
                noisy_gen_depth, target_depth, _ = add_noise(gen_depth, args, noise_scheduler, timesteps=timesteps)
                if cfg.data.use_depth_valid_only:
                    noisy_gen_depth[~depth_valid] = 0
                target.update({'depth': target_depth, 'depth_valid': depth_valid})
            gens_depth_with_mask = torch.cat([noisy_gen_depth, depth_valid], axis=2)
            data['gens'].update({'depth': gens_depth_with_mask, 'depth_valid': depth_valid})
    
    for mod in cond_mods:
        if mod == 'rgb':
            bs, nvc, nc, h, w = batch['cond_image'].shape
            # dinov2
            if 'dino' in batch:
                conds_rgb_dino = batch['dino']
            else:
                valid_cond_image, valid_indices = flatten_image_data(batch['cond_image'].to(weight_dtype), batch['mods_flags']['rgb'] == 0)
                valid_cond_image_vae, valid_indices = flatten_image_data(batch['gen_image'].to(weight_dtype), batch['mods_flags']['rgb'] == 0)
                if len(valid_indices[0]) > 0:
                    with torch.no_grad():
                        valid_conds_rgb_dino = feature_extractor(valid_cond_image, autoresize=False) # (num_valid, C, h, w)
                        valid_conds_rgb_vae = vae.encode(valid_cond_image_vae.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                        conds_rgb_dino = torch.zeros((bs, nvc) + valid_conds_rgb_dino.shape[1:], dtype=weight_dtype, device=device)
                        conds_rgb_vae = torch.zeros((bs, nvc) + valid_conds_rgb_vae.shape[1:], dtype=weight_dtype, device=device)
                        conds_rgb_dino[valid_indices] = valid_conds_rgb_dino
                        conds_rgb_vae[valid_indices] = valid_conds_rgb_vae
                        conds_rgb = torch.cat([conds_rgb_dino, conds_rgb_vae], dim=2)
                else:
                    size_down = batch['gen_image'].shape[-1] // 8
                    conds_rgb = torch.zeros(bs, nvc, 768+4, size_down, size_down, dtype=weight_dtype, device=device)
            data['conds'].update({'rgb': conds_rgb})
        elif mod == 'ray':
            conds_ray = batch['cond_rays'].clone()
            conds_ray[:, :, :3] = shift_scale_normalize(conds_ray[:, :, :3], cfg.data.shift_scales.ray.dirs[0], cfg.data.shift_scales.ray.dirs[1])
            conds_ray[:, :, 3:] = shift_scale_normalize(conds_ray[:, :, 3:], cfg.data.shift_scales.ray.moms[0], cfg.data.shift_scales.ray.moms[1])
            data['conds'].update({'ray': conds_ray})
        elif mod == 'depth':
            conds_depth = batch['cond_depth'].clone()
            depth_valid = (conds_depth > 0) & (conds_depth < 10)
            conds_depth = shift_scale_normalize(conds_depth, cfg.data.shift_scales.depth[0], cfg.data.shift_scales.depth[1])
            if cfg.data.use_depth_valid_only:
                conds_depth[~depth_valid] = 0
            conds_depth_with_mask = torch.cat([conds_depth, depth_valid], axis=2)
            data['conds'].update({'depth': conds_depth_with_mask})
        # text work only for conditions
        elif mod == 'global_caption' or mod == 'local_caption':
            prompt = batch[mod]
            if mod == 'local_caption':
                # for a fast encoding, we flatten the string list and recover it into matrices later
                prompt_flatten, prompt_positions = flatten_string_list(prompt) 
                prompt_positions = torch.from_numpy(np.array(prompt_positions))
            elif mod == 'global_caption':
                prompt_flatten = prompt
            
            if prompt_flatten == []:  # no captions for this time
                prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2 = [], [], [], []
            else:
                with torch.no_grad():
                    prompt_embeds_flatten, _, prompt_attention_mask_flatten, _ = \
                        hunyuan_encode_prompt(
                            prompt=prompt_flatten,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            device=device,
                            dtype=weight_dtype,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            max_sequence_length=77,
                            text_encoder_index=0,
                        )
                    prompt_embeds_2_flatten, _, prompt_attention_mask_2_flatten, _ = \
                        hunyuan_encode_prompt(
                            prompt=prompt_flatten,
                            tokenizer=tokenizer_2,
                            text_encoder=text_encoder_2,
                            device=device,
                            dtype=weight_dtype,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            max_sequence_length=256,
                            text_encoder_index=1,
                        )
                if mod == 'local_caption':
                    prompt_embeds = torch.zeros((bs, num_view) + prompt_embeds_flatten.shape[1:], dtype=prompt_embeds_flatten.dtype, device=device)
                    prompt_embeds[prompt_positions[:, 0], prompt_positions[:, 1]] = prompt_embeds_flatten
                    prompt_attention_mask = torch.zeros((bs, num_view) + prompt_attention_mask_flatten.shape[1:], dtype=prompt_attention_mask_flatten.dtype, device=device)
                    prompt_attention_mask[prompt_positions[:, 0], prompt_positions[:, 1]] = prompt_attention_mask_flatten
                    prompt_embeds_2 = torch.zeros((bs, num_view) + prompt_embeds_2_flatten.shape[1:], dtype=prompt_embeds_2_flatten.dtype, device=device)
                    prompt_embeds_2[prompt_positions[:, 0], prompt_positions[:, 1]] = prompt_embeds_2_flatten
                    prompt_attention_mask_2 = torch.zeros((bs, num_view) + prompt_attention_mask_2_flatten.shape[1:], dtype=prompt_attention_mask_2_flatten.dtype, device=device)
                    prompt_attention_mask_2[prompt_positions[:, 0], prompt_positions[:, 1]] = prompt_attention_mask_2_flatten
                elif mod == 'global_caption':
                    prompt_embeds = prompt_embeds_flatten.unsqueeze(1)
                    prompt_attention_mask = prompt_attention_mask_flatten.unsqueeze(1)
                    prompt_embeds_2 = prompt_embeds_2_flatten.unsqueeze(1)
                    prompt_attention_mask_2 = prompt_attention_mask_2_flatten.unsqueeze(1)
            data['conds'][mod] = {}
            data['conds'][mod].update({'prompt_embeds': prompt_embeds,
                                       'prompt_attention_mask': prompt_attention_mask,
                                       'prompt_embeds_2': prompt_embeds_2,
                                       'prompt_attention_mask_2': prompt_attention_mask_2})
    
    return data, target, timesteps


def draw_text(height, width, text, position=(20, 10), font_size=15, max_width=215):
    if text == "":
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    image = Image.new('RGB', (width, height), (0, 0, 0))  # black background
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size = font_size)
    x, y = position
    lines = []
    words = text.split()
    while words:
        line = ''
        while words:
            word = words[0]
            # get the size of the last word
            width, _ = draw.textbbox((0, 0), line + word, font=font)[2:]
            if width <= max_width:
                line = line + words.pop(0) + ' '
            else:
                break
        lines.append(line)
    
    # draw line by line
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255))  # white word
        y += draw.textbbox((0, 0), line, font=font)[3] 
    
    image = np.array(image)
    return image


def draw_val_groundtruth_images(val_data, cfg, task_name, device="cuda"):
    ## Groundtruth image
    H, W = cfg.train.val_height, cfg.train.val_width
    mods_flags = val_data[task_name]['mods_flags']
    cond_mods = [mod for mod in mods_flags if (mods_flags[mod] == 0).sum() > 0]
    gen_mods = [mod for mod in mods_flags if (mods_flags[mod] == 1).sum() > 0]
    gt_images = None
    batch_size, nv = val_data['view_id'].shape[:2]
    ## RGB
    if 'rgb' in cond_mods or 'rgb' in gen_mods:
        cond_image = val_data['cond_image'].flatten(0, 1)
        cond_image = F.interpolate(cond_image, (H, W), mode="bilinear").permute(0, 2, 3, 1) * 0.5 + 0.5
        gt_images = cond_image if gt_images is None else torch.cat([gt_images, cond_image], dim=2)  # concat gt and pred horizontally
    ## Ray
    if 'ray' in cond_mods or 'ray' in gen_mods:
        cond_rays = val_data['cond_rays']
        cond_rays = F.interpolate(cond_rays.flatten(0, 1), (H, W), mode="nearest").permute(0, 2, 3, 1)
        dirs, moms = vis_rays(cond_rays[..., :3], cond_rays[..., 3:], to_0_1=True)
        gt_images = torch.cat([dirs, moms], dim=2) if gt_images is None else torch.cat([gt_images, dirs, moms], dim=2)
    ## Depth
    if 'depth' in cond_mods or 'depth' in gen_mods:
        cond_depth = val_data['cond_depth'].flatten(0, 1)
        cond_depth /= cond_depth.reshape(batch_size * nv, -1).max(dim=1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_depth = F.interpolate(cond_depth, (H, W), mode="nearest").permute(0, 2, 3, 1)#.repeat(1, 1, 1, 3)
        # corlorful
        cond_depth = batch_convert_array_to_pil(cond_depth[..., 0])
        gt_images = cond_depth if gt_images is None else torch.cat([gt_images, cond_depth], dim=2)
    ## Local caption
    if 'local_caption' in cond_mods or 'local_caption' in gen_mods:
        flattened_local_caption = [item for sublist in val_data['local_caption'] for item in sublist]
        local_caption_img = torch.cat([torch.from_numpy(draw_text(H, W, 'local:' + caption))[None]/255. for caption in flattened_local_caption])
        gt_images = local_caption_img if gt_images is None else torch.cat([gt_images, local_caption_img.to(device)], dim=2)
    ## Global caption
    if 'global_caption' in cond_mods or 'global_caption' in gen_mods:
        flattened_global_caption = [item for item in val_data['global_caption'] for _ in range(nv)]
        global_caption_img = torch.cat([torch.from_numpy(draw_text(H, W, 'global:' + caption))[None]/255. for caption in flattened_global_caption])
        gt_images = global_caption_img if gt_images is None else torch.cat([gt_images, global_caption_img.to(device)], dim=2)

    ## final reshape
    gt_images = gt_images.reshape((batch_size, nv) + gt_images.shape[1:]).flatten(1, 2)   # [bs, h * nv, w * num_mods, 3] concat views vertically  

    return gt_images


def inference_ddpm_and_get_images(data, model, vae, val_noise_scheduler, dtype, device, num_inference_step, gt_image, cfg, guidance_scale=3.0):
    mmod_preds = inference_ddpm_call_varmod(
        model=model,
        scheduler=val_noise_scheduler,
        device=device,
        data=data,
        num_inference_steps=num_inference_step,
        guidance_scale=guidance_scale,
        cfg = cfg,
    )
    H, W = cfg.train.val_height, cfg.train.val_width
    mods_flags = data['mods_flags']
    cond_mods = [mod for mod in mods_flags if (mods_flags[mod] == 0).sum() > 0]
    gen_mods = [mod for mod in mods_flags if (mods_flags[mod] == 1).sum() > 0]
    bs, nv = mmod_preds['view_id'].shape[:2]
    pred_image = None
    ## default output
    pred_rgb, pred_ray, pred_depth = None, None, None
    rgb_mask, ray_mask, depth_mask = \
        torch.zeros([bs, nv], device=device).bool(), torch.zeros([bs, nv], device=device).bool(), torch.zeros([bs, nv], device=device).bool()
    ## RGB
    if 'rgb' in gen_mods:
        pred_rgb, rgb_mask = mmod_preds['gens']['rgb'], mmod_preds['gens']['rgb_mask'].reshape(-1)
        with torch.no_grad():
            pred_rgb = vae.decode(pred_rgb.to(dtype).flatten(0,1) / vae.config.scaling_factor, return_dict=False)[0]
        pred_rgb = (pred_rgb / 2 + 0.5).clamp(0, 1)
        pred_rgb[~rgb_mask] = 0  # only show valid predictions
        pred_rgb = F.interpolate(pred_rgb, (H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        pred_rgb = pred_rgb.reshape((bs, nv) + pred_rgb.shape[1:])    
        rgb_mask = rgb_mask.reshape((bs, nv) + rgb_mask.shape[1:])
        pred_image = pred_rgb if pred_image is None else torch.cat([pred_image, pred_rgb], dim=-2)  
    elif 'rgb' in cond_mods:
        pred_rgb = torch.zeros([bs, nv, H, W, 3], dtype=dtype).to(device=device) 
        pred_image = pred_rgb if pred_image is None else torch.cat([pred_image, pred_rgb], dim=-2)  
    ## Ray
    if 'ray' in gen_mods:      
        pred_ray, ray_mask = mmod_preds['gens']['ray'], mmod_preds['gens']['ray_mask'].reshape(-1)
        pred_ray = F.interpolate(pred_ray.flatten(0, 1), (H, W), mode="nearest").permute(0, 2, 3, 1)
        pred_ray = pred_ray.reshape((bs, nv) + pred_ray.shape[1:])
        ray_mask = ray_mask.reshape((bs, nv) + ray_mask.shape[1:])
        pred_dirs, pred_moms = vis_rays(pred_ray[..., :3], pred_ray[..., 3:], to_0_1=True)
        pred_dirs[~ray_mask.reshape(bs, nv)] = 0   # only show valid predictions
        pred_moms[~ray_mask.reshape(bs, nv)] = 0   # only show valid predictions
        pred_image = torch.cat([pred_dirs, pred_moms], dim=-2) if pred_image is None else torch.cat([pred_image, pred_dirs, pred_moms], dim=-2)
    elif 'ray' in cond_mods:
        pred_ray = torch.zeros([bs, nv, H, W * 2, 3], dtype=dtype).to(device=device)
        pred_image = pred_ray if pred_image is None else torch.cat([pred_image, pred_ray], dim=-2)
    ## Depth
    if 'depth' in gen_mods:
        pred_depth, depth_mask = mmod_preds['gens']['depth'], mmod_preds['gens']['depth_mask']
        pred_depth = F.interpolate(pred_depth.flatten(0, 1), (H, W), mode="nearest").permute(0, 2, 3, 1)#.repeat(1, 1, 1, 3)
        # corlorful
        pred_depth = batch_convert_array_to_pil(pred_depth[..., 0]).unflatten(0, (bs, nv))
        pred_depth[~depth_mask] = 0  # only show valid predictions
        pred_image = pred_depth if pred_image is None else torch.cat([pred_image, pred_depth], dim=-2) 
    elif 'depth' in cond_mods:
        pred_depth = torch.zeros([bs, nv, H, W, 3], dtype=dtype).to(device=device)
        pred_image = pred_depth if pred_image is None else torch.cat([pred_image, pred_depth], dim=-2)
    ## Local caption
    if 'local_caption' in cond_mods:
        empty = torch.zeros(bs, nv, H, W, 3, dtype=dtype, device=device)
        pred_image = torch.cat([pred_image, empty], dim=-2)
    ## Global caption
    if 'global_caption' in cond_mods:
        empty = torch.zeros(bs, nv, H, W, 3, dtype=dtype, device=device)
        pred_image = torch.cat([pred_image, empty], dim=-2)
    
    # concat groundtruth and prediction
    red_strip = torch.zeros((bs, 32) + (gt_image.shape[2:]), dtype=dtype, device=device)
    red_strip[..., 0] = 1
    np_image = (torch.cat([gt_image, red_strip, pred_image.flatten(1, 2)], dim=1).float().cpu().numpy() * 255).astype(np.uint8)
    
    return np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds


def model_inference(models, data_handler, view_ids, mod_flags, preprocessor, cfg, args, device, weight_dtype, guidance_scale=1.5, H=512, W=512, seed=None, vae_encoding=None):
    # parse models
    model, noise_scheduler, tokenizer, text_encoder, tokenizer_2, text_encoder_2, vae, feature_extractor = \
        models['model'], models['noise_scheduler'], models['tokenizer'], models['text_encoder'], models['tokenizer_2'], models['text_encoder_2'], models['vae'], models['feature_extractor']

    # process data
    eval_data = data_handler.select_via_indices(view_ids)
    tensor_recursive_to(eval_data, lambda x:x.to(device))
    eval_data = data_handler.mod_flags_update(eval_data, mod_flags) 
    data, target, _ = prepare_train_data(
        eval_data, preprocessor.gen_mods, preprocessor.cond_mods, noise_scheduler, feature_extractor, vae, weight_dtype, cfg, args, is_val=True,
        tokenizer=tokenizer, text_encoder=text_encoder, tokenizer_2=tokenizer_2, text_encoder_2=text_encoder_2)
    # NOTE: if provided vae encoding, replace with the prepared one to avoid applying vae encoder & decoder again
    # FIXME: This is a hard-code fix for single-to-3d generation because we found repeatedly using vae encoder & decoder would degrade the results
    # We replace the first values with the generated ones
    # For other tasks, this part could be ignored
    if vae_encoding is not None:
        num_in = vae_encoding.shape[1]
        data['conds']['rgb'][:, :num_in] = vae_encoding
    eval_data['data'] = data
    tensor_recursive_to(eval_data, lambda x: x.to(weight_dtype) if x.dtype == torch.float32 else x)
    
    # build and concat groundtruth images
    num_inference_step = cfg.eval.val_inference_steps
    mods_flags = data['mods_flags']
    cond_mods = [mod for mod in mods_flags if (mods_flags[mod] == 0).sum() > 0]
    gen_mods = [mod for mod in mods_flags if (mods_flags[mod] == 1).sum() > 0]
    gt_images = None
    batch_size, nv = eval_data['view_id'].shape[:2]
    ## RGB
    if 'rgb' in cond_mods or 'rgb' in gen_mods:
        cond_image = eval_data['cond_image'].flatten(0, 1)
        cond_image = F.interpolate(cond_image, (H, W), mode="bilinear").permute(0, 2, 3, 1) * 0.5 + 0.5
        gt_images = cond_image if gt_images is None else torch.cat([gt_images, cond_image], dim=2)  # concat gt and pred horizontally
    ## Ray
    if 'ray' in cond_mods or 'ray' in gen_mods:
        cond_rays = eval_data['cond_rays']
        cond_rays = F.interpolate(cond_rays.flatten(0, 1), (H, W), mode="nearest").permute(0, 2, 3, 1)
        dirs, moms = vis_rays(cond_rays[..., :3], cond_rays[..., 3:], to_0_1=True)
        gt_images = torch.cat([dirs, moms], dim=2) if gt_images is None else torch.cat([gt_images, dirs, moms], dim=2)
    ## Depth
    if 'depth' in cond_mods or 'depth' in gen_mods:
        cond_depth = eval_data['cond_depth'].flatten(0, 1)
        cond_depth /= cond_depth.reshape(batch_size * nv, -1).max(dim=1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_depth = F.interpolate(cond_depth, (H, W), mode="nearest").permute(0, 2, 3, 1)#.repeat(1, 1, 1, 3)
        # corlorful
        cond_depth = batch_convert_array_to_pil(cond_depth[..., 0])
        gt_images = torch.cat([gt_images, cond_depth], dim=2) if gt_images is None else torch.cat([gt_images, cond_depth], dim=2)
    ## Local caption
    if 'local_caption' in cond_mods or 'local_caption' in gen_mods:
        flattened_local_caption = [item for sublist in eval_data['local_caption'] for item in sublist]
        local_caption_img = torch.cat([torch.from_numpy(draw_text(H, W, 'local:' + caption))[None]/255. for caption in flattened_local_caption])
        gt_images = local_caption_img if gt_images is None else torch.cat([gt_images, local_caption_img.to(device)], dim=2)
    ## Global caption
    if 'global_caption' in cond_mods or 'global_caption' in gen_mods:
        flattened_global_caption = [item for item in eval_data['global_caption'] for _ in range(nv)]
        global_caption_img = torch.cat([torch.from_numpy(draw_text(H, W, 'global:' + caption))[None]/255. for caption in flattened_global_caption])
        gt_images = global_caption_img if gt_images is None else torch.cat([gt_images, global_caption_img.to(device)], dim=2)
    gt_images = gt_images.reshape((batch_size, nv) + gt_images.shape[1:]).flatten(1, 2)   # concat views vertically
    
    # inference
    data = eval_data['data']
    set_seed(seed)
    np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds = \
        inference_ddpm_and_get_images(
            data, model, vae, noise_scheduler, weight_dtype, device, num_inference_step, gt_images, cfg, guidance_scale
        )
    torch.cuda.empty_cache()
    
    return np_image, pred_rgb, rgb_mask, pred_ray, ray_mask, pred_depth, depth_mask, mmod_preds, eval_data