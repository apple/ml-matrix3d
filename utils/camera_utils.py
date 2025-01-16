#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import numpy as np
import torch
import json
import splines
import splines.quaternion
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.utils import opencv_from_cameras_projection


def fov_to_focal(fov, size):
    # convert fov angle in degree to focal
    return size / np.tan(fov * np.pi / 180.0 / 2.0) / 2.0


def focal_to_fov(focal, size):
    # convert focal to fov angle in degree
    return 2.0 * np.arctan(size / (2.0 * focal)) * 180.0 / np.pi


def set_pytorch3d_cameras_eye_at_up(azimuths, elevations, distance=1.0):
    nv = azimuths.shape[0]
    azimuths, elevations = np.deg2rad(azimuths), np.deg2rad(elevations)
    x = distance * np.sin(azimuths) * np.cos(elevations)
    y = distance * np.sin(elevations)
    z = distance * np.cos(azimuths) * np.cos(elevations) * -1
    
    at = torch.tensor([[0., 0., 0.]]).repeat(nv, 1).float()
    up = torch.tensor([[0., 1., 0.]]).repeat(nv, 1).float()
    eye = torch.tensor([x, y, z]).T.float()
    
    R, T = look_at_view_transform(eye=eye, at=at, up=up)
    
    return R, T



def fit_spline_given_pyt3d_cameras(pyt3d_camera, n_frames=80, scales=8, tension=0.5, 
                                          continuity=0.0, bias=0.0, is_loop=True):
    num_keyframes = len(pyt3d_camera)
    end_frame = num_keyframes if is_loop else num_keyframes - 1
    timestamps = np.linspace(0, end_frame, n_frames, endpoint=False, )
    quaternions_wxyz = matrix_to_quaternion(pyt3d_camera.R).numpy()
    positions = pyt3d_camera.get_camera_center().numpy()
    focals = pyt3d_camera.focal_length.numpy()
    orientation_spline = splines.quaternion.KochanekBartels(
        [
            splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(wxyz, shift=-1))
            for wxyz in quaternions_wxyz
        ],
        tcb=(tension, continuity, bias),
        endconditions="closed" if is_loop else "natural",
    )
    position_spline = splines.KochanekBartels(
        [position for position in positions],
        tcb=(tension, continuity, bias),
        endconditions="closed" if is_loop else "natural",
    )
    focal_spline = splines.KochanekBartels(
        [foc for foc in focals],
        tcb=(tension, continuity, bias),
        endconditions="closed" if is_loop else "natural",
    )
    quats = orientation_spline.evaluate(timestamps)
    quat_array = np.array([[quat.scalar, *quat.vector] for quat in quats], dtype=np.float32)
    points_array = position_spline.evaluate(timestamps).astype(np.float32)
    focal_array = focal_spline.evaluate(timestamps).astype(np.float32)
    
    # convert back to pyt3d
    R = quaternion_to_matrix(torch.from_numpy(quat_array))
    points = torch.from_numpy(points_array).float()
    T = torch.bmm(-R.permute(0, 2, 1), points[..., None])[..., 0]
    spline_focal = torch.from_numpy(focal_array)
    spline_p0 = pyt3d_camera.principal_point[0].unsqueeze(0).repeat(n_frames, 1)
    image_size = pyt3d_camera.image_size[0].unsqueeze(0).repeat(n_frames, 1)
    
    # scale the cameras based on the scales
    if scales == 1:
        scales_values = torch.Tensor([1.0 + 0.0 * s for s in range(scales)])
    elif scales == 2:
        scales_values = torch.Tensor([1.0 + 0.05 * s for s in range(scales)])
    elif scales == 3:
        scales_values = torch.Tensor([0.8 + 0.2 * s for s in range(scales)])
    elif scales == 8:
        scales_values = torch.Tensor([0.9 + 0.05 * s for s in range(scales)])
    else:
        raise NotImplementedError("Unsupported number of scales for spline fitting. Please configure it manually.")
    R_matrices = R[None].repeat(scales, 1, 1, 1)
    T_matrices = T[None].repeat(scales, 1, 1) * scales_values.unsqueeze(-1).unsqueeze(-1).repeat(1, n_frames, 1)
    

    new_R_matrices = []
    new_T_matrices = []
    from scipy.spatial.transform import Rotation
    for i in range(scales):
        new_T = T_matrices[i]
        # quat = Rotation.from_matrix(R_matrices[i].cpu().numpy()).as_quat()
        # rotation_matrix = Rotation.from_quat(quat).as_matrix()
        new_R_matrices.append(R_matrices[i])
        new_T_matrices.append(new_T)
    new_R_matrices = torch.stack(new_R_matrices).flatten(0, 1)
    new_T_matrices = torch.stack(new_T_matrices).flatten(0, 1)
    
    
    spline_focal = spline_focal.repeat(scales, 1)
    spline_p0 = spline_p0.repeat(scales, 1)
    image_size = image_size.repeat(scales, 1)
    
    spline_cam = PerspectiveCameras(
        R=new_R_matrices,
        T=new_T_matrices,
        focal_length=spline_focal,
        principal_point=spline_p0,
        image_size=image_size,
        device=R.device,
    )
    return spline_cam


def write_pyt3d_camera_to_nerfstudio_json(folder, ref_camera, gen_camera, eval_camera=None, has_ply=False, has_mask=False, has_depth=False):
    # train jsons
    transform = {}
    frames_list = []
    # reference_cameras
    num_ref_frames = len(ref_camera)
    camera_centers = ref_camera.get_camera_center()
    R_cv_w2c, tvec_cv, Ks = opencv_from_cameras_projection(ref_camera, image_size=ref_camera.image_size)
    for i in range(num_ref_frames):
        frame = {}
        R_c2w = ref_camera.R[i]
        R_c2w_blender = R_c2w.clone()
        # convert pytorch3d camera to blender/opengl camera
        R_c2w_blender[:, [0, 2]] *= -1  
        # R_c2w = R_cv_w2c[i]#.T
        T_c2w = camera_centers[i].unsqueeze(-1)
        c2w = torch.cat([R_c2w_blender, T_c2w], dim=-1)
        c2w_homo = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]])]).float()
        frame["file_path"] = f"images/ref_frame_{i:04d}.png"
        frame["transform_matrix"] = c2w_homo.tolist()
        frame["fl_x"] = Ks[i][0, 0].item()
        frame["fl_y"] = Ks[i][1, 1].item()
        frame["cx"] = Ks[i][0, 2].item()
        frame["cy"] = Ks[i][1, 2].item()
        frame["w"] = ref_camera.image_size[0, 1].item()
        frame["h"] = ref_camera.image_size[0, 0].item()
        if has_mask:
            frame["mask_path"] = f"masks/ref_frame_{i:04d}.png"
        if has_depth:
            frame["depth_file_path"] = f"depths/ref_frame_{i:04d}.npy"
        frames_list.append(frame)
    # generation cameras
    num_gen_frames = len(gen_camera)
    camera_centers = gen_camera.get_camera_center()
    R_cv_w2c, tvec_cv, Ks = opencv_from_cameras_projection(gen_camera, image_size=gen_camera.image_size)
    for i in range(num_gen_frames):
        frame = {}
        R_c2w = gen_camera.R[i]
        R_c2w_blender = R_c2w.clone()
        # convert pytorch3d camera to blender/opengl camera
        R_c2w_blender[:, [0, 2]] *= -1  
        # R_c2w = R_cv_w2c[i]#.T
        T_c2w = camera_centers[i].unsqueeze(-1)
        c2w = torch.cat([R_c2w_blender, T_c2w], dim=-1)
        c2w_homo = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]])]).float()
        frame["file_path"] = f"images/frame_{i:04d}.png"
        frame["transform_matrix"] = c2w_homo.tolist()
        frame["fl_x"] = Ks[i][0, 0].item()
        frame["fl_y"] = Ks[i][1, 1].item()
        frame["cx"] = Ks[i][0, 2].item()
        frame["cy"] = Ks[i][1, 2].item()
        frame["w"] = gen_camera.image_size[0, 1].item()
        frame["h"] = gen_camera.image_size[0, 0].item()
        if has_mask:
            frame["mask_path"] = f"masks/frame_{i:04d}.png"
        if has_depth:
            frame["depth_file_path"] = f"depths/frame_{i:04d}.npy"
        frames_list.append(frame)
    transform["frames"] = frames_list
    if has_ply:
        transform["ply_file_path"] = "ref_pred_pointcloud.ply"
    with open(os.path.join(folder, 'transforms_train.json'), 'w') as json_file:
        json.dump(transform, json_file, indent=4)
    
    # test jsons
    if eval_camera is not None:
        transform = {}
        frames_list = []
        # evaluation_cameras
        num_eval_frames = len(eval_camera)
        camera_centers = eval_camera.get_camera_center()
        R_cv_w2c, tvec_cv, Ks = opencv_from_cameras_projection(eval_camera, image_size=eval_camera.image_size)
        for i in range(num_eval_frames):
            frame = {}
            R_c2w = eval_camera.R[i]
            R_c2w_blender = R_c2w.clone()
            # convert pytorch3d camera to blender/opengl camera
            R_c2w_blender[:, [0, 2]] *= -1  
            # R_c2w = R_cv_w2c[i]#.T
            T_c2w = camera_centers[i].unsqueeze(-1)
            c2w = torch.cat([R_c2w_blender, T_c2w], dim=-1)
            c2w_homo = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]])]).float()
            frame["file_path"] = f"images/eval_frame_{i:04d}.png"
            frame["transform_matrix"] = c2w_homo.tolist()
            frame["fl_x"] = Ks[i][0, 0].item()
            frame["fl_y"] = Ks[i][1, 1].item()
            frame["cx"] = Ks[i][0, 2].item()
            frame["cy"] = Ks[i][1, 2].item()
            frame["w"] = eval_camera.image_size[0, 1].item()
            frame["h"] = eval_camera.image_size[0, 0].item()
            if has_mask:
                frame["mask_path"] = f"masks/eval_frame_{i:04d}.png"
            if has_depth:
                frame["depth_file_path"] = f"depths/eval_frame_{i:04d}.npy"
            frames_list.append(frame)
        transform["frames"] = frames_list
        with open(os.path.join(folder, 'transforms_test.json'), 'w') as json_file:
            json.dump(transform, json_file, indent=4)
        