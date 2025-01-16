#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Adapted from code originally written by David Novotny.
"""

import ipdb  # noqa: F401
import torch
from pytorch3d.transforms import Rotate, Translate


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    if p_intersect is None:
        return None, None, None, None
    _, p_line_intersect = point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(
        dim=-1
    )
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)

    # I_eps = torch.zeros_like(I_min_cov.sum(dim=-3)) + 1e-10
    # p_intersect = torch.pinverse(I_min_cov.sum(dim=-3) + I_eps).matmul(sum_proj)[..., 0]
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    # I_min_cov.sum(dim=-3): torch.Size([1, 1, 3, 3])
    # sum_proj: torch.Size([1, 1, 3, 1])

    # p_intersect = np.linalg.lstsq(I_min_cov.sum(dim=-3).numpy(), sum_proj.numpy(), rcond=None)[0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        return None, None
        ipdb.set_trace()
        assert False
    return p_intersect, r


def point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def compute_optical_axis_intersection(cameras):
    centers = cameras.get_camera_center()
    principal_points = cameras.principal_point

    one_vec = torch.ones((len(cameras), 1), device=centers.device)
    optical_axis = torch.cat((principal_points, one_vec), -1)

    # optical_axis = torch.cat(
    #     (principal_points, cameras.focal_length[:, 0].unsqueeze(1)), -1
    # )

    pp = cameras.unproject_points(optical_axis, from_ndc=True, world_coordinates=True)
    pp2 = torch.diagonal(pp, dim1=0, dim2=1).T

    directions = pp2 - centers
    centers = centers.unsqueeze(0).unsqueeze(0)
    directions = directions.unsqueeze(0).unsqueeze(0)

    p_intersect, p_line_intersect, _, r = intersect_skew_line_groups(
        p=centers, r=directions, mask=None
    )

    if p_intersect is None:
        dist = None
    else:
        p_intersect = p_intersect.squeeze().unsqueeze(0)
        dist = (p_intersect - centers).norm(dim=-1)

    return p_intersect, dist, p_line_intersect, pp2, r

class IntersectionException(Exception):
    pass

def normalize_cameras(cameras, scale=1.0, add_cameras=False):
    """
    Normalizes cameras such that the optical axes point to the origin, the rotation is
    identity, and the norm of the translation of the first camera is 1.

    Args:
        cameras (pytorch3d.renderer.cameras.CamerasBase).
        scale (float): Norm of the translation of the first camera.

    Returns:
        new_cameras (pytorch3d.renderer.cameras.CamerasBase): Normalized cameras.
        undo_transform (function): Function that undoes the normalization.
    """

    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = (
        new_cameras.get_world_to_view_transform()
    )  # potential R is not valid matrix
    p_intersect, dist, p_line_intersect, pp, r = compute_optical_axis_intersection(
        cameras
    )

    if p_intersect is None:
        raise IntersectionException

    d = dist.squeeze(dim=1).squeeze(dim=0)[0]
    # Degenerate case
    if d == 0:
        # print(cameras.T)
        # print(new_transform.get_matrix()[:, 3, :3])
        raise IntersectionException

    # Can't figure out how to make scale part of the transform too without messing up R.
    # Ideally, we would just wrap it all in a single Pytorch3D transform so that it
    # would work with any structure (eg PointClouds, Meshes).
    tR = Rotate(new_cameras.R[0].unsqueeze(0)).inverse()
    tT = Translate(p_intersect)
    t = tR.compose(tT)

    new_transform2 = t.compose(new_transform)    # = t.get_matrix() @ new_transform.get_matrix()
    new_cameras.R = new_transform2.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform2.get_matrix()[:, 3, :3] / d * scale
    
    scene_scale = scale / d

    def undo_transform(cameras):
        cameras_copy = cameras.clone()
        cameras_copy.T *= d / scale
        new_t = (
            t.inverse().compose(cameras_copy.get_world_to_view_transform()).get_matrix()
        )
        cameras_copy.R = new_t[:, :3, :3]
        cameras_copy.T = new_t[:, 3, :3]
        return cameras_copy

    if add_cameras:
        return new_cameras, undo_transform, scene_scale, t
    else:
        return new_cameras, undo_transform, scene_scale


def normalize_cameras_inference(cameras, scale=1.0):
    """
    Normalizes cameras such that the optical axes point to the origin, the rotation is
    identity, and the norm of the translation of the first camera is 1.

    Args:
        cameras (pytorch3d.renderer.cameras.CamerasBase).
        scale (float): Norm of the translation of the first camera.

    Returns:
        new_cameras (pytorch3d.renderer.cameras.CamerasBase): Normalized cameras.
        undo_transform (function): Function that undoes the normalization.
    """

    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = (
        new_cameras.get_world_to_view_transform()
    )  # potential R is not valid matrix

    # Can't figure out how to make scale part of the transform too without messing up R.
    # Ideally, we would just wrap it all in a single Pytorch3D transform so that it
    # would work with any structure (eg PointClouds, Meshes).
    tR = Rotate(new_cameras.R[0].unsqueeze(0)).inverse()
    d = new_cameras.T[0].norm()
    T = torch.Tensor([0., 0., 1.]) - new_cameras.T[0]
    tT = Translate(T.unsqueeze(0))
    t = tR.compose(tT)

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3] / d * scale

    def undo_transform(cameras):
        cameras_copy = cameras.clone()
        cameras_copy.T *= d / scale
        new_t = (
            t.inverse().compose(cameras_copy.get_world_to_view_transform()).get_matrix()
        )
        cameras_copy.R = new_t[:, :3, :3]
        cameras_copy.T = new_t[:, 3, :3]
        return cameras_copy

    return new_cameras, undo_transform


def first_camera_transform(cameras, rotation_only=True):
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()
    tR = Rotate(new_cameras.R[0].unsqueeze(0))
    if rotation_only:
        t = tR.inverse()
    else:
        tT = Translate(new_cameras.T[0].unsqueeze(0))
        t = tR.compose(tT).inverse()

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3]

    return new_cameras
