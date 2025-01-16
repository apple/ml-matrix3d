#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision import transforms
from PIL import Image
import os
import cv2
import copy
from pytorch3d.renderer import PerspectiveCameras
from model.utils.rays import cameras_to_rays
from model.utils.normalize import normalize_cameras, IntersectionException


def blender_to_pytorch3d(
    camera_matrix: torch.Tensor, 
    R: torch.Tensor, 
    T: torch.Tensor, 
    image_size: torch.Tensor,
    ) -> PerspectiveCameras:
    # modified from function: pytorch3d.utils.cameras_from_opencv_projection
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # we first convert RT to opencv coords and then to pytorch3d
    R_cv, T_cv = R.clone(), T.clone()
    R_cv[:, 1:] *= -1
    T_cv[:, 1:] *= -1
    
    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R_cv.clone().permute(0, 2, 1)
    T_pytorch3d = T_cv.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=R.device,
    )


class Preprocessor():
    def __init__(self, cfg, fov=60.0):
        super().__init__()
        # load config
        cfg_eval = copy.deepcopy(cfg.data)
        if 'evaluation_overwrite' in cfg_eval:
            overwrite = cfg_eval['evaluation_overwrite']
            for key in overwrite:
                cfg_eval[key] = overwrite[key]
        self.cfg = cfg
        self.configs = cfg_eval
        self.fov = fov
        
        # modality configs
        self.modalities = self.configs['modalities']
        self.mod_probs = torch.tensor(self.configs.modalities_probs).float()
        self.mod_probs /= self.mod_probs.sum(dim=1, keepdim=True)
        self.mod_probs[:, -1] = 1 - self.mod_probs[:, :-1].sum(dim=1)
        self.gen_mods = [self.modalities[i] for i, prob in enumerate(self.mod_probs) if prob[0] > 0]
        self.cond_mods = [self.modalities[i] for i, prob in enumerate(self.mod_probs) if prob[1] > 0] 
        self.used_mods = list(set(self.gen_mods) | set(self.cond_mods)) 
        self.max_view = self.configs.num_view[-1]
        self.default_modalities_mapping = {
            'rgb': {
                'cond_image': torch.zeros(self.max_view, 3, self.configs.cond_size, self.configs.cond_size, dtype=torch.float32),
                'gen_image': torch.zeros(self.max_view, 3, self.configs.gen_size, self.configs.gen_size, dtype=torch.float32),
            },
            'ray': {
                'extrinsic': torch.zeros(self.max_view, 3, 4, dtype=torch.float32),
                'intrinsic': torch.zeros(self.max_view, 3, 3, dtype=torch.float32),
                'cond_rays': torch.zeros(self.max_view, 6, self.configs.raymap_size, self.configs.raymap_size, dtype=torch.float32),
                'gen_rays': torch.zeros(self.max_view, 6, self.configs.raymap_size, self.configs.raymap_size, dtype=torch.float32),
            },
            'depth': {
                'depth': torch.zeros(self.max_view, self.configs.depth_size, self.configs.depth_size, dtype=torch.float32),
                'cond_depth': torch.zeros(self.max_view, 1, self.configs.depth_size, self.configs.depth_size, dtype=torch.float32),
                'gen_depth': torch.zeros(self.max_view, 1, self.configs.depth_size, self.configs.depth_size, dtype=torch.float32),
            },
            'local_caption': {
                'local_caption': ['' for _ in range(self.max_view)],
            },
            'global_caption': {
                'global_caption': '',
            },
        }

    @staticmethod
    def assert_all_same_shape(*lists):
        for L in lists:
            shapes = [getattr(x, 'shape', None) for x in L]
            if not all(s is None for s in shapes):
                first = next(s for s in shapes if s is not None)
                if any(s != first and s is not None for s in shapes):
                    raise AssertionError("Within each list, all shapes must match or all be None.")

    @staticmethod
    def load_image(img_path):
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = image[:, :, [2, 1, 0, 3]]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        return image
    
    @staticmethod
    def load_depth_co3dv2(depth_path):
        if not os.path.exists(depth_path):
            return None
        else:
            # co3d offical depth loading
            with Image.open(depth_path) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
                depth = (
                    np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                    .astype(np.float32)
                    .reshape((depth_pil.size[1], depth_pil.size[0]))
                )
            # filter out invalid depth values
            depth = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.)
            return depth
      
    @staticmethod
    def load_extrinsics(extrinsic_file):
        '''Load camera extrinsics (3x4)'''
        if not os.path.exists(extrinsic_file):
            return None
        extrinsic = np.array(np.loadtxt(extrinsic_file), ndmin=1)
        if len(extrinsic) == 12:
            extrinsic = extrinsic.reshape(3, 4)
        elif len(extrinsic) == 16:
            extrinsic = extrinsic.reshape(4, 4)[:3]
        else:
            raise ValueError(f'Invalid extrinsic file format: {extrinsic_file}. Supported 12-dim or 16-dim only.')
        return extrinsic

    @staticmethod
    def load_intrinsic(camera_file, h, w):
        if not os.path.exists(camera_file):
            return None
        camera_params = np.array(np.loadtxt(camera_file), ndmin=1)
        if len(camera_params) == 4:
            fx, fy, cx, cy = camera_params
        elif len(camera_params) == 1:
            focal = min(h, w) / np.tan(camera_params[0] * np.pi / 180.0 / 2.0) / 2.0
            fx, fy = focal, focal
            cx, cy = w / 2.0, h / 2.0
        else:
            raise ValueError(f'Invalid camera file format: {camera_file}. Supported 4-dim or 1-dim only.')
        K = np.array([
            [fx,  0,  cx],
            [ 0, fy,  cy],
            [ 0,  0,   1]
        ])
        return K
    
    @staticmethod
    def image_camera_preprocess(img, intrinsic=None, depth=None, out_size=512, depth_out_size=128):
        '''image & camera intrinsic center crop preprocess.
        '''
        height, width = img.shape[:2]
        square_length = int(min(width, height))
        
        # crop parameters without jitter
        left = (width - square_length) // 2
        top = (height - square_length) // 2
        right = left + square_length
        bottom = top + square_length
        img = img[top : bottom, left : right]
        
        # finally resize the image to out_size
        img = cv2.resize(img, (out_size, out_size), cv2.INTER_CUBIC)
        
        # modify the intrinsic if provided
        if intrinsic is not None:
            intrinsic[0, 2] -= left
            intrinsic[1, 2] -= top
            intrinsic[:2] *= out_size / square_length

        if depth is not None:
            depth = depth[top : bottom, left : right]
            depth = cv2.resize(depth, (depth_out_size, depth_out_size), interpolation=cv2.INTER_NEAREST)

        return img, intrinsic, depth
    
    @staticmethod
    def _blend_background(image, bg_color, channel_first=True, enable=True):
        # image: [0-1] array/tensor
        # bg_color: str or [0-1] float
        color_map = {
            'white': 1.0,
            'black': 0.0,
            'gray': 0.5,
        }
        num_channel = image.shape[1]
        if enable and num_channel == 4:
            if bg_color in color_map:
                bg_color = color_map[bg_color]
            if channel_first:
                return image[:,:3,:,:] * image[:,3:4,:,:] + bg_color * (1 - image[:,3:4,:,:])
            else:
                return image[:,:,:,:3] * image[:,:,:,3:4] + bg_color * (1 - image[:,:,:,3:4])
        else:
            return image[:,:3,:,:] if channel_first else image[:,:,:,:3]
    
    @staticmethod
    def gen_raymap_from_camera(intrinsic=None, extrinsic=None, pyt3d_camera=None, num_patches_x=32, num_patches_y=32, h=1024, w=1024, use_plucker=True, return_tensor=True):
        if pyt3d_camera is None:
            pyt3d_camera = blender_to_pytorch3d(
                camera_matrix=intrinsic, 
                R=extrinsic[:, :, :3], 
                T=extrinsic[:, :, 3],
                image_size=torch.Tensor([[h, w]])
            )
        rays = cameras_to_rays(
            cameras=pyt3d_camera,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            use_half_pix=True,
            crop_parameters=None,
            use_plucker=use_plucker,
        )
        if return_tensor:
            rays_patch = rays.rays.permute(0, 2, 1)
            rays_patch = rays_patch.reshape(rays_patch.shape[:2] + (num_patches_x, num_patches_y))
            return rays_patch
        else:
            return rays

    @staticmethod
    def pad_tensors(tensor_dict, length=0, keys=None, value=-1, except_keys=['num_views', 'view_id','scene_id']):
        if length == 0:
            raise ValueError('length must be positive')
        if keys is None:
            keys = tensor_dict.keys()
        for key in keys:
            if key in except_keys: continue
            tensor = tensor_dict[key]
            if type(tensor) == torch.Tensor:
                if tensor.shape[0] < length:
                    pad_shape = (length - tensor.shape[0],) + tensor.shape[1:]
                    tensor = torch.cat([tensor, torch.ones(pad_shape) * value], dim=0)
                    tensor_dict[key] = tensor
        return tensor_dict

    @staticmethod
    def pad_modalities(tensor_dict, modalities_mapping, max_view=None):
        for mod in modalities_mapping.keys():
            for key in modalities_mapping[mod]:
                if key not in tensor_dict:
                    tensor_dict[key] = modalities_mapping[mod][key]
        return tensor_dict
    
    @staticmethod
    def fov_to_intrinsic(fov, size=512.0):
        focal = size / np.tan(fov * np.pi / 180.0 / 2.0) / 2.0
        intrinsic = torch.Tensor([
            [focal,      0.0000,   size / 2.0],
            [  0.0000,    focal,   size / 2.0],
            [  0.0000,   0.0000,       1.0000],
        ])
        return intrinsic  

    def preprocess(self, image, config, extrinsic=None, intrinsic=None, height=None, width=None, depth=None, mask=None, num_view=8, indentity_camera_index=0, 
                   pad=True, pad_value=-1, pad_length=8, return_camera=False):
        ''' Preprocess the image, and generate the ray maps from the camera.
        Args:
            image: [b, h, w, 3/4] in [0, 255]
            extrinsic: [b, 3, 4]
            intrinsic: [b, 3, 3]
        '''
        out = {}
        # if image provided
        if image is not None:
            batch_size, height, width, _ = image.shape
            # RGB image
            gen_transform = T.Compose(
                [
                    T.Lambda(lambda arr:arr.permute(0,3,1,2)),
                    T.Resize(config.gen_size, antialias=True),
                    T.Lambda(lambda arr:arr/255.),
                    T.Lambda(lambda arr:self._blend_background(arr, config.background_color, enable=config.use_background)),
                    T.Normalize([0.5], [0.5]),
                ]
            )
            cond_transform = transforms.Compose(
                [
                    T.Lambda(lambda arr:arr.permute(0,3,1,2)),
                    T.Resize(config.cond_size, antialias=True),
                    T.Lambda(lambda arr:arr/255.),
                    T.Lambda(lambda arr:self._blend_background(arr, config.background_color, enable=config.use_background)),
                    T.Normalize([0.5], [0.5]),
                ]
            )  
            gen_image, cond_image = gen_transform(image), cond_transform(image)
            out.update(
                {
                'cond_image': cond_image, 
                'gen_image': gen_image,
                }
            )
            if mask is not None:
                gen_mask_transform = T.Compose(
                [
                    T.Lambda(lambda arr:arr.permute(0,3,1,2)),
                    T.Resize(config.gen_size, antialias=True),
                    T.Lambda(lambda arr:arr/255.),
                ]
                )
                cond_mask_transform = transforms.Compose(
                    [
                        T.Lambda(lambda arr:arr.permute(0,3,1,2)),
                        T.Resize(config.cond_size, antialias=True),
                        T.Lambda(lambda arr:arr/255.),
                    ]
                ) 
                gen_mask, cond_mask = gen_mask_transform(mask), cond_mask_transform(mask)
                out.update(
                    {
                    'cond_mask': cond_mask, 
                    'gen_mask': gen_mask,
                    }
                )
            
        # scene preprocessing, including poses (ray maps) and depths if available
        ## TODO: this preprocessing code needs to be refactored
        if 'ray' in self.modalities or 'depth' in self.modalities:
            # if ray provided
            if extrinsic is not None and intrinsic is not None:
                nan_mask = torch.isnan(extrinsic).sum(dim=-1).sum(dim=-1) > 0
                if nan_mask.sum() > 0:
                    trans = torch.randn(nan_mask.sum(), 3, dtype=extrinsic.dtype, device=extrinsic.device)
                    identity_RT = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device)[:3][None].repeat(nan_mask.sum(), 1, 1)
                    identity_RT[:, :3, 3] = trans
                    extrinsic[nan_mask] = identity_RT
                extrinsic = torch.nan_to_num(extrinsic)
                # convert to pytorch3d cameras
                pyt3d_camera = blender_to_pytorch3d(
                    camera_matrix=intrinsic[:num_view], 
                    R=extrinsic[:num_view, :, :3], 
                    T=extrinsic[:num_view, :, 3],
                    image_size=torch.Tensor([[height, width]]),
                )
                pyt3d_camera_original = pyt3d_camera.clone()
                # compute scene scale
                scaling_finish = False
                if config.dataset_type == 'object-centric':
                    # use optical axis intersection as scene center
                    try:
                        pyt3d_camera, undo_transform, scene_scale = normalize_cameras(pyt3d_camera, scale=1.0)
                        if depth is not None:
                            valid_depth = depth > 0
                            depth[valid_depth] *= scene_scale
                        scaling_finish = True
                    except IntersectionException:
                        print("Intersection degenerate, use fall back solution. Ignore this error when input unposed images.")
                if config.dataset_type == 'scenes' or not scaling_finish:
                    if depth is not None:
                        # normalize the scene according to the first view depth
                        # example dataset: hypersim
                        # set the median depth to 1.0
                        first_view_depth = depth[0]
                        median_depth = torch.median(first_view_depth[first_view_depth > 0])
                        scene_scale = 1.0 / median_depth
                        # re-scale depth map
                        valid_depth = depth > 0
                        depth[valid_depth] *= scene_scale
                        pyt3d_camera.T *= scene_scale
                    else:
                        # normalize the scene according to the camera positions only
                        # example dataset: realestate10k
                        # set the maximum camera distance to ref camera to 1.0
                        camera_positions = pyt3d_camera.get_camera_center()
                        camera_center = camera_positions[0]
                        distances = (camera_positions - camera_center).norm(dim=-1)
                        max_distance = distances.max()
                        scene_scale = 1.0 / max_distance 
                        pyt3d_camera.T *= scene_scale
                        
                if 'ray' in self.modalities:
                    # applied scene-scale to pytorch3d cameras and normalze to identity location
                    pyt3d_camera = self.normalize_cameras_to_identity(pyt3d_camera, indentity_camera_index=indentity_camera_index)
                    gt_w2c = pyt3d_camera.get_world_to_view_transform().get_matrix()
                    rays = self.gen_raymap_from_camera(
                        pyt3d_camera=pyt3d_camera, num_patches_x=self.configs.raymap_size, 
                        num_patches_y=self.configs.raymap_size, h=height, w=width, use_plucker=config.use_plucker)  

                    if pad:
                        pad_shape = (pad_length - rays.shape[0],) + rays.shape[1:]
                        rays = torch.cat([rays, torch.ones(pad_shape) * pad_value], dim=0)
                    rays = torch.nan_to_num(rays)
                    cond_rays, gen_rays = rays, rays
                    out.update(
                        {
                            'cond_rays': cond_rays, 
                            'gen_rays': gen_rays,
                            'gt_w2c': gt_w2c,
                            'gt_pyt3d_camera': pyt3d_camera,
                            'original_camera': pyt3d_camera_original,
                            'scene_scale': torch.Tensor([scene_scale]),
                        }
                    )
            # if ray is not provided but only depth is provided
            # this means we only need the depth as input or output
            # we only need to normalize the depth values according to the first depth
            else:
                if depth is not None:
                    first_view_depth = depth[0]
                    median_depth = torch.median(first_view_depth[first_view_depth > 0])
                    scene_scale = 1.0 / median_depth
                    # re-scale depth map
                    valid_depth = depth > 0
                    depth[valid_depth] *= scene_scale
            
            # save depth
            if depth is not None:
                # depth_indices, depth_values = self.depth_sampling(depth, num_samples_per_images=config.depth_samples_per_images)
                # for vae sample only now
                if config.per_sample_aug_enable:
                    rotates, scales = config.per_sample_aug.depth.rotate, config.per_sample_aug.depth.scale
                    value_scales = config.per_sample_aug.depth.value_scales
                    # first scale values
                    random_value_scales = np.random.uniform(value_scales[0], value_scales[1], num_view).astype(np.float32)[:, None, None]
                    depth *= random_value_scales
                    # then aug images (rotate and crop and scale)
                    depth_transform = T.Compose(
                        [
                            T.RandomRotation(degrees=rotates, interpolation=T.InterpolationMode.NEAREST),
                            T.RandomResizedCrop(
                                size=config.depth_size, 
                                scale=scales,                         
                                interpolation=T.InterpolationMode.NEAREST),

                        ]
                    )
                    depth = depth_transform(depth)
                    
                # to disparity
                disparity = depth.clone()
                disparity[depth > 0] = 1 / depth[depth > 0]
                disparity_values = disparity.unsqueeze(1)
                # depth_valid_mask = depth_values > 0
                out.update(
                    {
                        'cond_depth': disparity_values,
                        'gen_depth': disparity_values,
                    }
                )
        if return_camera:
            return out, pyt3d_camera
        else:
            return out

    @staticmethod
    def normalize_cameras_to_identity(pyt3d_camera, indentity_camera_index=0):
        # normalize the cameras to Rotation=I, Translation=[0,0,1]
        # In-place operations
        identity_RT = torch.Tensor([[1.,  0.,  0.,  0.],
                                    [0.,  1.,  0.,  0.],
                                    [0.,  0.,  1.,  0.],
                                    [0.,  0.,  1.,  1.]])
        # convert first RT to indentity
        pyt3d_w2c = pyt3d_camera.get_world_to_view_transform().get_matrix()
        transform_matrix = identity_RT @ pyt3d_w2c[indentity_camera_index].inverse() @ pyt3d_w2c
        pyt3d_camera.R = transform_matrix[:, :3, :3]
        pyt3d_camera.T = transform_matrix[:, 3, :3] 
        
        return pyt3d_camera

    @staticmethod
    def unsqueeze_batch_data(batch):
        # unsqueeze batch data to (batch_size, *)
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].unsqueeze(0)
            elif isinstance(batch[key], PerspectiveCameras):
                batch[key] = [batch[key]]
            elif isinstance(batch[key], dict):
                for sub_key in batch[key].keys():
                    batch[key][sub_key] = batch[key][sub_key].unsqueeze(0)
        return batch

    def __getitem_single_view__(self, path):
        ### 1. load data
        assert path.endswith(('.png', '.jpg'))
        # load images & cameras
        scene_id = os.path.basename(path)[:-4]
        image = self.load_image(path)
        intrinsic = self.load_intrinsic(path[:-4] + '.txt', h=image.shape[0], w=image.shape[1])
        image, intrinsic, _ = self.image_camera_preprocess(image, intrinsic, out_size=512)

        ### 2. preprocess
        height, width = image.shape[:2]
        image = torch.from_numpy(image).unsqueeze(0)
        if intrinsic is not None:
            intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
        else:
            # set default camera for single-view image
            focal = 512 / np.tan(self.fov * np.pi / 180.0 / 2.0) / 2.0
            intrinsic = torch.Tensor([
                [focal,      0.0000,   256.0000],
                [  0.0000,    focal,   256.0000],
                [  0.0000,   0.0000,     1.0000],
            ]).unsqueeze(0) 
        extrinsic = torch.eye(4).unsqueeze(0)
        transform = T.Compose(
            [
                T.Lambda(lambda arr:arr.permute(0,3,1,2)),
                T.Resize(self.configs.gen_size, antialias=True),
                T.Lambda(lambda arr:arr/255.),
                T.Lambda(lambda arr:self._blend_background(arr, "white", enable=True)),
                T.Normalize([0.5], [0.5]),
            ]
        )
        cond_transform = transforms.Compose(
            [
                T.Lambda(lambda arr:arr.permute(0,3,1,2)),
                T.Resize(self.configs.cond_size, antialias=True),
                T.Lambda(lambda arr:arr/255.),
                T.Lambda(lambda arr:self._blend_background(arr, "white", enable=True)),
                T.Normalize([0.5], [0.5]),
            ]
        )  
        gen_image, cond_image = transform(image), cond_transform(image)
        pyt3d_camera = blender_to_pytorch3d(
            camera_matrix=intrinsic, 
            R=extrinsic[:, :3, :3], 
            T=extrinsic[:, :3, 3],
            image_size=torch.Tensor([[height, width]]),
        )
        pyt3d_camera = self.normalize_cameras_to_identity(pyt3d_camera, indentity_camera_index=0)
        rays = self.gen_raymap_from_camera(
            pyt3d_camera=pyt3d_camera[0], num_patches_x=self.configs.raymap_size, 
            num_patches_y=self.configs.raymap_size, h=height, w=width, use_plucker=self.configs.use_plucker)   

        depth = torch.zeros(self.max_view, 1, self.configs.depth_size, self.configs.depth_size, dtype=torch.float32)
        cond_rays, gen_rays = rays, rays
        cond_depth, gen_depth = depth, depth

        torch_sample = {
            'cond_image': cond_image, 
            'gen_image': gen_image,
            'cond_rays': cond_rays, 
            'gen_rays': gen_rays,
            'cond_depth': cond_depth,
            'gen_depth': gen_depth,
            'height': torch.Tensor([height]),
            'width': torch.Tensor([width]),
            'intrinsic': intrinsic,
            'extrinsic': extrinsic,
            'gt_pyt3d_camera': pyt3d_camera,
            'mods_flags':
                {
                    'ray': torch.zeros([self.max_view]), 
                    'rgb': torch.ones([self.max_view]),
                    'depth': torch.ones([self.max_view]) * -1, 
                    'local_caption': torch.ones([self.max_view]) * -1,
                    'global_caption': torch.ones([self.max_view]) * -1,
                },
            'num_views': torch.Tensor([1]),
            'view_id': torch.arange(self.max_view),  # fixed views
            'scene_id': scene_id,            
        }
        # pad views and modalities at final
        torch_sample = self.pad_tensors(torch_sample, length=self.max_view, except_keys=['num_views', 'view_id','scene_id', 'mods_flags'])
        torch_sample = self.pad_modalities(torch_sample, self.default_modalities_mapping, max_view=self.max_view)
        torch_sample = self.unsqueeze_batch_data(torch_sample)
         
        return torch_sample
    
    def __getitem_multi_view__(self, root):
        ### 1. load data
        image_paths = sorted([f for f in os.listdir(root) if f.endswith(('.png','.jpg')) and not 'depth' in f])
        image_list, intrinsic_list, depth_list, extrinsic_list = [], [], [], []
        # load images & cameras
        for image_path in image_paths:
            image_id = image_path.split('.')[-2]
            image = self.load_image(os.path.join(root, image_path))
            intrinsic = self.load_intrinsic(
                os.path.join(root, f'{image_id}.txt'), h=image.shape[0], w=image.shape[1])
            # NOTE: this demo preprocessor only works with co3dv2 depth images!
            # Please modify the depth loading function to your self-defined depth images for other datasets.
            depth = self.load_depth_co3dv2(os.path.join(root, f'{image_id}_depth.png'))
            image, intrinsic, depth = self.image_camera_preprocess(image, intrinsic, depth, out_size=512, depth_out_size=self.configs.depth_size)
            extrinsic = self.load_extrinsics(os.path.join(root, f'{image_id}_ext.txt'))
            image_list.append(image)
            intrinsic_list.append(intrinsic)
            depth_list.append(depth)
            extrinsic_list.append(extrinsic)
        # all shape should match or all be None!
        self.assert_all_same_shape(image_list, intrinsic_list, depth_list, extrinsic_list)
        
        ### 2. preprocess (CO3DV2 dataset way)
        images = torch.from_numpy(np.stack(image_list))
        num_view = len(images)
        if intrinsic_list[0] is not None:
            intrinsics = torch.from_numpy(np.stack(intrinsic_list)).float()
        else:
            # set default camera for single-view image
            focal = 512 / np.tan(self.fov * np.pi / 180.0 / 2.0) / 2.0
            intrinsics = torch.Tensor([
                [focal,      0.0000,   256.0000],
                [  0.0000,    focal,   256.0000],
                [  0.0000,   0.0000,     1.0000],
            ]).unsqueeze(0).float().repeat(num_view, 1, 1)
        extrinsics = torch.from_numpy(np.stack(extrinsic_list)).float() if extrinsic_list[0] is not None else None
        depths = torch.from_numpy(np.stack(depth_list)) if depth_list[0] is not None else None
        scene_id = root.split('/')[-1]
        height, width = images[0].shape[:2]
        # scene normalization & camera normalization
        # NOTE: this actually wouldn't affect the inference but only the training and visualization comparisons with groundtruths
        # The default demo assumes that the scene is 'object-centric' while you could change this to 'scenes' for your own data
        # The preprocess fuction includes all pre-process procedures for used six training datasets during training
        torch_sample = self.preprocess(
            images, 
            self.configs, 
            extrinsic=extrinsics, 
            intrinsic=intrinsics,
            height=height,
            width=width,
            depth=depths,
            mask=None,
            num_view=num_view,
            pad_length=self.max_view,
            indentity_camera_index=0,   # set first-view camera as identity camera
        )
        # set default rays under unposed cases
        if extrinsic_list[0] is None:
            extrinsic = torch.eye(4).unsqueeze(0).repeat(num_view, 1, 1)
            pyt3d_camera = blender_to_pytorch3d(
                camera_matrix=intrinsics, 
                R=extrinsic[:, :3, :3], 
                T=extrinsic[:, :3, 3],
                image_size=torch.Tensor([[height, width]]),
            )
            pyt3d_camera = self.normalize_cameras_to_identity(pyt3d_camera, indentity_camera_index=0)
            rays = self.gen_raymap_from_camera(
                pyt3d_camera=pyt3d_camera, num_patches_x=self.configs.raymap_size, 
                num_patches_y=self.configs.raymap_size, h=height, w=width, use_plucker=self.configs.use_plucker)
            rays[1:] = -1  # we only need the first view
            torch_sample['cond_rays'], torch_sample['gen_rays'] = rays, rays
        # update other data
        torch_sample.update({
            'height': torch.Tensor([height]),
            'width': torch.Tensor([width]),
            'mods_flags':
                {
                    'ray': torch.zeros([self.max_view]), 
                    'rgb': torch.ones([self.max_view]),
                    'depth': torch.ones([self.max_view]) * -1, 
                    'local_caption': torch.ones([self.max_view]) * -1,
                    'global_caption': torch.ones([self.max_view]) * -1,
                },  # set default task as rgb -> pose
            'num_views': torch.Tensor([num_view]),
            'view_id': torch.arange(self.max_view),  # fixed views
            'scene_id': scene_id,
        })
        # pad views and modalities at final
        torch_sample = self.pad_tensors(torch_sample, length=self.max_view, except_keys=['num_views', 'view_id','scene_id', 'mods_flags'])
        torch_sample = self.pad_modalities(torch_sample, self.default_modalities_mapping, max_view=self.max_view)
        torch_sample = self.unsqueeze_batch_data(torch_sample)
        
        return torch_sample
        
 
    def __call__(self, data_path, input_type):
        if input_type == 'single-view':
            return self.__getitem_single_view__(data_path)      
        elif input_type == 'multi-view':
            return self.__getitem_multi_view__(data_path)