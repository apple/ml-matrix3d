#! /bin/sh
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname "$SCRIPT_DIR")
export NERFSTUDIO_METHOD_CONFIGS="splatfacto_matrix3d=splatfacto_matrix3d.splatfacto_configs:splatfacto_method"
export PYTHONPATH=$PYTHONPATH:$REPO_DIR

EXP_NAME=$1
INPUT_PATH=$2
NAME_EXT=$(basename "$INPUT_PATH")
NAME="${NAME_EXT%.*}"

### Step 1: Generation: Create novel view observations
CUDA_VISIBLE_DEVICES=0 python pipeline_unposed_few_shot_to_3d.py \
    --config configs/config_stage3.yaml \
    --exp_name $EXP_NAME \
    --data_path $INPUT_PATH \
    --spline_scales 3 \
    --num_samples 80 \
    --num_depth_runs_for_init_depth 15 \
    --checkpoint_path checkpoints/matrix3d_512.pt \
    --mixed_precision fp16 \
    --random_seed 1 \
    --use_loop_traj 1 \
    --dataset arkitscenes


### Step 2: Reconstruction: 3DGS optimization
cd results/$EXP_NAME/$NAME

# 1. optimization
ITERS=3000
NUM_IMG=5
ns-train splatfacto_matrix3d \
    --data transforms_train.json \
    --mixed-precision False \
    --output-dir outputs \
    --timestamp exps \
    --viewer.quit-on-train-completion True \
    --max-num-iterations $ITERS \
    --steps-per-save 1000 \
    --pipeline.model.num-downscales 0 \
    --pipeline.model.resolution-schedule 500 \
    --pipeline.datamanager.max-num-iterations $ITERS \
    --pipeline.datamanager.num_image_each_iteration $NUM_IMG \
    --pipeline.model.warmup-length 500 \
    --pipeline.model.densify-grad-thresh 0.0008 \
    --pipeline.model.cull-alpha-thresh 0.2 \
    --pipeline.model.cull-scale-thresh 0.5 \
    --pipeline.model.cull-screen-size 0.5 \
    --pipeline.model.reset-alpha-every 15 \
    --pipeline.model.refine-every 100 \
    --pipeline.model.use_scale_regularization True \
    --pipeline.model.max-gauss-ratio 6 \
    --pipeline.model.apply-annealing False \
    --pipeline.model.stop-screen-size-at 4000 \
    --pipeline.model.stop-split-at 2000 \
    --pipeline.model.sh-degree 3 \
    --pipeline.model.sh-degree-interval 800 \
    --pipeline.model.accumulation-lambda 0.5 \
    --pipeline.model.full-accumulation-lambda 5.0 \
    --pipeline.model.start-full-accumulation 1500 \
    --pipeline.model.ssim-lambda 0.2 \
    --pipeline.model.lpips-lambda 20.0 \
    --pipeline.model.l1-lambda-on-captured-views 20.0 \
    --pipeline.model.l1-lambda-on-generation-views 1.0 \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.use-absgrad True \
    --pipeline.model.lpips-downsample 4 \
    --pipeline.model.lpips-min-img-size 256 \
    --pipeline.model.lpips-patch-size 512 \
    --pipeline.model.lpips-no-resize True \
    --pipeline.model.depth-l1-lambda 10.0 \
    --pipeline.model.depth-ranking-lambda 20.0 \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.use-bilateral-grid False \
    nerfstudio-data  --center-method none --orientation-method none --auto-scale-poses False --train-split-fraction 1.0 --load-3D-points True --depth-unit-scale-factor 1.0
# 2. use ns-render to render frames
ns-render dataset --load-config outputs/splatfacto_matrix3d/exps/config.yml --image-format png --split=train --output-path renders
# 3. write frames into videos
python $REPO_DIR/utils/write_videos.py --render_root renders --type scene --num_splines 3