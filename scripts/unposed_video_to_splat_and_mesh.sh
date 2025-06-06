#! /bin/sh

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname "$SCRIPT_DIR")
export NERFSTUDIO_METHOD_CONFIGS="splatfacto_matrix3d=splatfacto_matrix3d.splatfacto_configs:splatfacto_method,nerfacto_matrix3d=nerfacto_matrix3d.nerfacto_configs:nerfacto_method"
export PYTHONPATH=$PYTHONPATH:$REPO_DIR

EXP_NAME=$1
INPUT_PATH=$2
NAME_EXT=$(basename "$INPUT_PATH")
NAME="${NAME_EXT%.*}"

python utils/video_to_images.py --data $INPUT_PATH --output_folder results/$EXP_NAME/converted_images --num_frames_target 7

### Step 1: Generation: Create 360-degree novel views
CUDA_VISIBLE_DEVICES=0 python pipeline_unposed_few_shot_to_3d.py \
    --config configs/config_stage3.yaml \
    --exp_name $EXP_NAME \
    --data_path results/$EXP_NAME/converted_images \
    --spline_scales 1 \
    --num_samples 40 \
    --num_cond_images 7 \
    --num_depth_runs_for_init_depth 5 \
    --checkpoint_path checkpoints/matrix3d_512.pt \
    --mixed_precision fp16 \
    --random_seed 1 \
    --use_loop_traj 1 \
    --dataset co3dv2

### Step 2: Reconstruction: 3DGS optimization
cd results/$EXP_NAME

# 1. optimization
ITERS=3000
NUM_IMG=5
ns-train splatfacto_matrix3d \
    --data . \
    --mixed-precision False \
    --output-dir outputs/ \
    --timestamp exps \
    --viewer.quit-on-train-completion True \
    --max-num-iterations $ITERS \
    --steps-per-save 1000 \
    --pipeline.model.num-downscales 0 \
    --pipeline.model.resolution-schedule 500 \
    --pipeline.model.warmup-length 500 \
    --pipeline.model.densify-grad-thresh 0.0008 \
    --pipeline.model.cull-alpha-thresh 0.2 \
    --pipeline.model.cull-scale-thresh 0.5 \
    --pipeline.model.cull-screen-size 0.5 \
    --pipeline.model.reset-alpha-every 15 \
    --pipeline.model.refine-every 100 \
    --pipeline.model.use_scale_regularization True \
    --pipeline.model.max-gauss-ratio 6 \
    --pipeline.model.stop-screen-size-at 4000 \
    --pipeline.model.stop-split-at 2000 \
    --pipeline.model.sh-degree 3 \
    --pipeline.model.sh-degree-interval 800 \
    --pipeline.model.ssim-lambda 0.2 \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.use-absgrad True \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.use-bilateral-grid False \
    nerfstudio-data  --center-method none --orientation-method none --auto-scale-poses False --train-split-fraction 1.0 --load-3D-points True --depth-unit-scale-factor 1.0

ns-export gaussian-splat --load-config outputs/nerfacto_matrix3d/exps/config.yml --output-dir gaussian_splat_export

ns-train nerfacto_matrix3d \
    --data . \
    --mixed-precision False \
    --output-dir outputs/ \
    --timestamp exps \
    --viewer.quit-on-train-completion True \
    --max-num-iterations $ITERS \
    --pipeline.model.predict-normals True \
    --steps-per-save 1000 \
    nerfstudio-data  --center-method none --orientation-method none --auto-scale-poses False --train-split-fraction 1.0 --load-3D-points True --depth-unit-scale-factor 1.0


ns-export tsdf --load-config outputs/nerfacto_matrix3d/exps/config.yml --output-dir export_mesh --resolution 256 --target_num_faces 1000000
##python nerfstudio/scripts/texture.py --load-config CONFIG.yml --input-mesh-filename FILENAME --output-dir OUTPUT_DIR