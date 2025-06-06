# Matrix3D Gaussian Splat Exporter with Normal Support

This module provides tools for exporting Matrix3D models to Gaussian Splat format with proper normal support.

## Key Features

- **Normal Export**: Correctly calculates and exports normal vectors for each Gaussian
- **Multiple Export Methods**: Supports different approaches to export models:
  - Direct checkpoint loading (most reliable)
  - Nerfstudio's exporter with proper normal calculations
- **Robust Error Handling**: Falls back to alternative methods when one fails

## Command Line Usage

You can use the exporter directly from the command line:

```bash
python export_splat.py --input /path/to/checkpoint/or/config \
                     --output-dir output_directory \
                     --output-filename splat.ply \
                     --normal-method calculate \
                     --ply-color-mode sh_coeffs
```

### Arguments

- `--input`: Path to model checkpoint or config
- `--output-dir`: Directory to save exported files
- `--output-filename`: Name of output file (default: splat.ply)
- `--normal-method`: Method to generate normals ("calculate", "zero", or "model")
- `--ply-color-mode`: Color mode ("sh_coeffs" or "rgb")

## Programmatic Usage

```python
from utils.export.direct_export import export_matrix3d

result = export_matrix3d(
    "/path/to/checkpoint.pt",
    "/output/directory",
    "output.ply"
)

if result:
    print(f"Export successful: {result}")
else:
    print("Export failed")
```

## Normal Calculation Methods

- **calculate**: Derives normals from Gaussian orientations (quaternions)
- **model**: Uses model's predicted normals if available
- **zero**: Uses zero vectors for normals

## Troubleshooting

If you encounter errors during export:

1. Check that the input path points to a valid checkpoint or config file
2. Ensure you have the necessary dependencies installed
3. Try a different export method or normal calculation method

## Contributing

Contributions to improve the exporter are welcome. Please follow the project's contribution guidelines.
