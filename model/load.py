#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import safetensors
import torch
import torch.utils.checkpoint
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import HunyuanDiTPipeline

from model.feature_extractors import SpatialDino
from model.dit import DiT

def load_model(cfg, checkpoint_path, device='cuda:0', weight_dtype=torch.float16):
    # Load scheduler, tokenizer and models.
    if cfg.eval.scheduler == "DDPM":
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.scheduler_url, subfolder="scheduler")
    elif cfg.eval.scheduler == "DDIM":
        noise_scheduler = DDIMScheduler.from_pretrained(cfg.model.scheduler_url, subfolder="scheduler")
    
    # Freeze vae and text_encoder
    feature_extractor = SpatialDino(
        freeze_weights=True, 
        model_type="dinov2_vitb14",
        num_patches_x=cfg.modalities.rgb.width,
        num_patches_y=cfg.modalities.rgb.width,
    )
    hunyuan_pipe = HunyuanDiTPipeline.from_pretrained(cfg.model.decoder_url, torch_dtype=torch.float16)
    tokenizer, text_encoder = hunyuan_pipe.tokenizer, hunyuan_pipe.text_encoder
    tokenizer_2, text_encoder_2 = hunyuan_pipe.tokenizer_2, hunyuan_pipe.text_encoder_2
    vae = hunyuan_pipe.vae
    del hunyuan_pipe
    
    # Build model and load from checkpoint
    cfg.used_modalities = {key: cfg.modalities[key] for key in ['rgb', 'ray', 'depth', 'local_caption', 'global_caption']}
    model = DiT(modalities=cfg.used_modalities, **cfg.model)
    if os.path.splitext(checkpoint_path)[-1] == '.safetensors':
        state_dict = safetensors.torch.load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path)['module']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()
    print('Loaded model from:', checkpoint_path)
    print('missing_keys', missing_keys)
    print('unexpected_keys', unexpected_keys)
    
    # Move non-trainables and cast to weight_dtype
    vae.to(device, dtype=weight_dtype)
    feature_extractor.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2.to(device, dtype=weight_dtype)
    model.to(device, dtype=weight_dtype)
    
    # Package all components into one dict
    models = {
        'model': model,
        'noise_scheduler': noise_scheduler,
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'tokenizer_2': tokenizer_2,
        'text_encoder_2': text_encoder_2,
        'vae': vae,
        'feature_extractor': feature_extractor
    }
    
    return models