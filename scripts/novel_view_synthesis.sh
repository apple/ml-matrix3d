#! /bin/sh
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
data_path=$1

CUDA_VISIBLE_DEVICES=0 python pipeline_novel_view_synthesis.py \
    --config configs/config_stage3.yaml \
    --data_path $data_path \
    --mixed_precision fp16 \
    --guidance_scale 1.5 \
    --checkpoint_path checkpoints/matrix3d_512.pt
