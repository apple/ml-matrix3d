# Matrix3D: Large Photogrammetry Model All-in-One
Yuanxun Lu<sup>1\*</sup>, Jingyang Zhang<sup>2\*</sup>, Tian Fang<sup>2</sup>, Jean-Daniel Nahmias<sup>2</sup>, Yanghai Tsin<sup>2</sup>, Long Quan<sup>3</sup>, Xun Cao<sup>1</sup>, Yao Yao<sup>1†</sup>, Shiwei Li<sup>2</sup>  
<sup>1</sup>Nanjing University, <sup>2</sup>Apple, <sup>3</sup>HKUST  
<sup>\*</sup>Equal contribution <sup>†</sup>Corresponding author

### [Project Page](https://nju-3dv.github.io/projects/matrix3d/) | [Paper](https://arxiv.org/abs/2502.07685) | [Weights](#environment-setup) 

This is the official implementation of Matrix3D, a unified model that performs several photogrammetry subtasks, including pose estimation, depth prediction, and novel view synthesis using the same model.

This repository includes the model inference pipeline and the modified 3DGS reconstruction pipeline for 3D reconstruction.

<p align="center">
  <img width="90%" src="docs/inference-pipe.png"/>
</p>
<p align="center">
   <em>Matrix3D supports various photogrammetry tasks via masked inference.</em>
<br>

## Environment Setup

- This project is successfully tested on Ubuntu 20.04 with PyTorch 2.4 (Python 3.10). We recommend creating a new environment and install necessary dependencies:

  ```
  conda create -y -n matrix3d python=3.10
  conda activate matrix3d
  # Here we take Pytorch 2.4 with cuda 11.8 as an example
  # If you install a different PyTorch version, please select a matched xformers/pytorch3d version
  pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
  pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.7+pt2.4.0cu118
  pip install -r requirements.txt
  # fixed the requirement conflicts from nerfstudio
  pip install timm==1.0.11
  ```
  Some dependencies may require CUDA with the same version used by `torch` in your system, and the installation may not work out of the box. Please refer to their official repo for troubleshooting.

* Download the Pre-trained model:
  * Download the checkpoints: [matrix3d_512.pt](https://ml-site.cdn-apple.com/models/matrix3d/matrix3d_512.pt)
  * Create a `checkpoints` folder and put the pre-trained model into it.
* (Optional) Download `IS-Net` checkpoint if you would like to use single-view to 3d reconstruction:
  * Download the pre-trained model `isnet-general-use.pth` from the [DIS official repo](https://github.com/xuebinqin/DIS) and also put it into the `checkpoints` folder.

## Run Demo

- Matrix3D supports several photogrammetry tasks and their dynamic compositions via masked inference. Here we provide several example scripts on the CO3Dv2 dataset. All results will be saved to the `results` folder by default.

  - **Novel View Synthesis**

    ```
    sh scripts/novel_view_synthesis.sh examples/co3dv2-samples/31_1359_4114
    ```

    This script demonstrates the usage of novel view synthesis from single-view image input. 

    For all diffusion sampling tasks, we use indicators `mod_flags` and `view_ids` to control the input states in `L48-L56`. You could try to set a different modality flag or view numbers to achieve different tasks, such as predict novel views from 2 posed RGB images.

  - **Pose Estimation**

    ```
    sh scripts/pose_estimation.sh examples/co3dv2-samples/31_1359_4114
    ```

    This script demonstrates the usage of pose prediction from images. The saved `*.png` and `*.html` file demonstrates a visual comparison between predictions and groundtruth values.

    Replace the data root to an unposed data folder like `examples/unposed-samples/co3dv2/201_21613_43652` would generate the results without comparisons to groundtruth poses. 

    It is **strongly recommended** to provide the camera intrinsics saved in the .txt files since the model is trained with known camera intrinsics. If not, the processor would set a default fov=60 and performance may degrade. You could also change the default Fov value by passing `--default_fov`. 

  - **Depth Prediction**

    ```
    sh scripts/depth_prediction.sh examples/co3dv2-samples/31_1359_4114
    ```

    This script demonstrates the usage of depth prediction from several posed images. The back-projected groundtruth and prediction point clouds can be found in the folder.

- By dynamically combining the above tasks, one could later apply a modified 3DGS pipeline to achieve 3D reconstruction from various inputs, even with unknown camera parameters. In the following, we provide two specific examples:

  - **Single-view to 3D**

    ```
    sh scripts/single_view_to_3d.sh single-view-to-3d examples/single-view/skull.png
    ```

    The 3DGS rendering results are saved in `results/single-view-to-3d/skull/3DGS-render-traj.mp4`.

    In this task, camera Fov is set to 60 by default, while you could also manually set it by creating a `$name.txt` file along with the image. The dataprocessor would automatically load it. For example, you could replace the `skull.png` with `ghost.png`. 

    Please check the `examples/single-view` folder for more examples.

  - **Unposed Few-shot to 3D**

    ```
    sh scripts/unposed_fewshot_to_3d_co3dv2.sh unposed-fewshot-to-3d examples/unposed-samples/co3dv2/31_1359_4114
    ```

    This script demonstrates a reconstruction process from unposed images in CO3Dv2 dataset. Note that the camera trajectories of novel views are sampled on fitted splines from predicted poses and designed to work under object-centric scenes. The specific interpolation video is saved as `3DGS-render-traj1.mp4` by default. You could also change to apply reconstruction on arkitscenes data as follows:
    
    ```
    sh scripts/unposed_fewshot_to_3d_arkitscenes.sh unposed-fewshot-to-3d examples/unposed-samples/arkitscenes/41069043
    ```
    
    The only difference lies in the splined camera generation while the 3DGS part is exactly same. You may need to tune the parameters of trajectory generation and 3DGS reconstruction for different datasets to achieve higher performance.

- Based on the examples above, you can flexibly define specifically tailored tasks by combining different inputs.

- Notes:

  - When trying on the diffusion process, please carefully assign the values of indicators `mods_flags` and `view_ids`. Besides, the model is trained with a maximum view number of 8, so do not set `view_ids` larger than 8 views.
  - The example data in `examples/co3dv2-samples` and `examples/unposed-samples` are part of CO3Dv2 and ARKitScenes datasets. The camera extrinsic is saved in FOV values or Blender camera coordinates. In processing, we would convert them into PyTorch3D cameras, and these part codes could be found in `L654-659` from `data/data_preprocessor.py`. Therefore, it is easy for users to change to different camera representations, e.g., you could apply the official Pytorch3D conversion function `pytorch3d.utils.cameras_from_opencv_projection` to convert OpenCV cameras into Pytorch3D cameras.


## License

This sample code is released under the [LICENSE](LICENSE) terms.

## Citation
```
@article{lu2025matrix3d,
  title={Matrix3D: Large Photogrammetry Model All-in-One},
  author={Lu, Yuanxun and Zhang, Jingyang and Fang, Tian and Nahmias, Jean-Daniel and Tsin, Yanghai and Quan, Long and Cao, Xun and Yao, Yao and Li, Shiwei},
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
