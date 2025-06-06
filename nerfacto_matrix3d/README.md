# Nerfacto Matrix3D

This module provides an integration between Nerfacto model and Matrix3D data format, enabling high-quality neural reconstruction from Matrix3D generated data.

## Features

- Seamless integration with Matrix3D data pipeline
- Normal prediction for improved geometry reconstruction
- Depth supervision for more accurate depth estimation
- Camera distance-based loss weighting for balancing real vs. generated views
- Optimized for 3D reconstruction from both captured and generated views

## Usage

```bash
# Set environment variables
export NERFSTUDIO_METHOD_CONFIGS="nerfacto_matrix3d=nerfacto_matrix3d.nerfacto_configs:nerfacto_method"
export PYTHONPATH=$PYTHONPATH:/path/to/repo

# Run training
ns-train nerfacto_matrix3d \
    --data transforms_train.json \
    --mixed-precision False \
    --output-dir outputs \
    --timestamp exps \
    --viewer.quit-on-train-completion True \
    --max-num-iterations 1500 \
    --steps-per-save 1000 \
    --pipeline.model.predict-normals True \
    --pipeline.model.output-depth-during-training True \
    nerfstudio-data  --load-3D-points True
```

## Parameters

### Matrix3D Specific Parameters

- `depth_l1_lambda`: Weight for L1 depth supervision loss
- `depth_ranking_lambda`: Weight for depth ranking loss
- `l1_lambda_on_captured_views`: Weight for L1 image loss on captured views
- `l1_lambda_on_generation_views`: Weight for L1 image loss on generated views
- `output_depth_during_training`: Whether to output depth during training

## Export Options

The trained model can be exported using various methods:

```bash
# Export mesh using Poisson Surface Reconstruction
python utils/export/export_cli.py \
    --type poisson \
    --load-config outputs/nerfacto_matrix3d/exps/config.yml \
    --output-dir poisson_export \
    --normal-method model_output
```

## Integration with Pipeline

Nerfacto Matrix3D is designed to work seamlessly within the Matrix3D pipeline, bridging Matrix3D data with Nerfacto's neural reconstruction capabilities.
