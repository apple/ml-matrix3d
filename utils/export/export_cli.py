import argparse
import sys
import os
import pathlib

# Import your patched gaussian splat exporter
import utils.export.patched_exporter as exporter

# Optionally: import/define your own export functions for other types, or import from nerfstudio
# e.g.
# from nerfstudio.scripts.exporter import export_pointcloud, export_tsdf, export_poisson, export_marching_cubes, export_cameras

def main():
    parser = argparse.ArgumentParser(description="Patched ns-export multiplexer")
    parser.add_argument("--type", required=True, choices=[
        "pointcloud", "tsdf", "poisson", "marching-cubes", "cameras", "gaussian-splat"
    ], help="Export type (same as ns-export --type)")
    parser.add_argument("--load-config", required=True, type=pathlib.Path, help="Input checkpoint or config")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path, help="Output directory")
    parser.add_argument("--output-filename", type=str, default=None, help="Optional output file name (PLY/PNG/...)")
    parser.add_argument("--normal-method", type=str, default="calculate", help="Normals: calculate | zero | model")
    parser.add_argument("--ply-color-mode", type=str, default="sh_coeffs", help="PLY color mode for gaussian-splat")
    # Add any other extra arguments here for other export types

    args, unknown = parser.parse_known_args()

    # Dispatch based on export type
    if args.type == "gaussian-splat":
        exporter.ExportGaussianSplat(
            load_config=args.load_config,
            output_dir=args.output_dir
        ).main()
        print("[✓] Gaussian splat export with normal support complete.")
    elif args.type == "pointcloud":
        # Example: import or call your variant, or fallback to original tool
        exporter.ExportPointCloud(
            load_config=args.load_config,
            output_dir=args.output_dir
        ).main()
    elif args.type == "tsdf":
        exporter.ExportTSDFMesh(
            load_config=args.load_config,
            output_dir=args.output_dir
        ).main()
    elif args.type == "poisson":
        exporter.ExportPoissonMesh(
            load_config=args.load_config,
            output_dir=args.output_dir,
            normal_method="open3d"
        ).main()
    elif args.type == "marching-cubes":
        exporter.ExportMarchingCubesMesh(
            load_config=args.load_config,
            output_dir=args.output_dir
        ).main()
    elif args.type == "cameras":
        exporter.ExportCameraPoses(
            load_config=args.load_config,
            output_dir=args.output_dir,
        ).main()
    else:
        print(f"[!] Unsupported export type: {args.type}")
        sys.exit(1)

if __name__ == "__main__":
    main()