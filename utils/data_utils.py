#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import numpy as np
import os
import cv2
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.implicitron.tools.point_cloud_utils import (
    render_point_cloud_pytorch3d,
    get_rgbd_point_cloud,
)


MOD_FLAG_TABLE = {
    'c': 0,
    'g': 1,
    'x': -1,
}

class DataHandler():
    '''DataHandler for multi-view multi-modal data'''
    def __init__(self, data: dict, pad_length=None, except_keys=None):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")
        self.data = data
        self.batch_size, self.num_view_raw = data['view_id'].shape
        if pad_length:
            self.pad_batch_data_using_first_value(pad_length, except_keys)


    def pad_batch_data_using_first_value(self, 
                                         target_length, 
                                         except_keys=['scene_id', 'global_caption', 'num_views', 'train_ids', 'test_ids', 'scene_scale']):
        # pad every value to target length
        for key in self.data.keys():
            if key in except_keys: continue  
            elif type(self.data[key]) == dict:
                if key == 'mods_flags':
                    # use -1 (not used flag) for all mod flags
                    for sub_key in self.data[key].keys():
                        current_length = self.data[key][sub_key].size(1)
                        padding_size = target_length - current_length
                        padding = torch.ones([1])[None].repeat(self.batch_size, padding_size) * -1
                        self.data[key][sub_key] = torch.cat([self.data[key][sub_key], padding], dim=1)
                else:
                    raise NotImplementedError()
            elif isinstance(self.data[key], torch.Tensor):
                current_length = self.data[key].size(1)
                if current_length < target_length:
                    padding_size = target_length - current_length
                    first_value = self.data[key][:, :1, ...] 
                    padding = first_value.repeat(1, padding_size, *[1] * (self.data[key].dim() - 2))
                    self.data[key] = torch.cat([self.data[key], padding], dim=1)
            elif isinstance(self.data[key], list):
                for i in range(len(self.data[key])):
                    if isinstance(self.data[key][i], list):
                        current_length = len(self.data[key][i])
                        self.data[key][i].extend([self.data[key][i][0] for _ in range(target_length - current_length)])
                    elif isinstance(self.data[key][i], PerspectiveCameras): 
                        current_length = len(self.data[key][i])
                        padding_size = target_length - current_length
                        indices = [k for k in range(current_length)] + [0 for j in range(padding_size)]
                        self.data[key][i] = self.data[key][i][indices]
                        # hard code pass pytorch3d camera
                    elif isinstance(self.data[key][i], str): continue
                        # hard code pass global caption   
                    else: raise NotImplementedError(f'meet type {type(self.data[key])} not implemented! key={key}')
    

    def select_via_indices(self, 
                           indices=np.array([0, 1]), 
                           except_keys=['scene_id', 'global_caption', 'num_views', 'train_ids', 'test_ids', 'scene_scale'], 
                           reset_viewid=True):
        new_data = {}
        for key, value in self.data.items():
            if key in except_keys:
                new_data[key] = value
            elif isinstance(value, dict):
                if key == 'mods_flags':
                    new_data[key] = {}
                    for sub_key in value.keys():
                        new_data[key][sub_key] = value[sub_key][:, indices].clone()
                else:
                    raise NotImplementedError()
            elif isinstance(value, torch.Tensor):
                new_data[key] = value[:, indices].clone()
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, list):
                        new_list.append([item[idx] for idx in indices])
                    elif isinstance(item, PerspectiveCameras):
                        new_list.append(item[indices.tolist()])
                    elif isinstance(item, str):
                        new_list.append(item)
                    else:
                        raise NotImplementedError(f'meet type {type(item)} not implemented! key={key}')
                new_data[key] = new_list
            elif isinstance(value, bool):
                new_data[key] = value
            else:
                raise NotImplementedError(f'meet type {type(value)} not implemented! key={key}')
            
        if reset_viewid and 'view_id' in new_data:
            bs, num_view = new_data['view_id'].shape
            new_data['view_id'] = torch.arange(num_view)[None].repeat(bs, 1)
        return new_data
    
    @staticmethod
    def mod_flags_update(batch, mod_flags):
        num_view = batch['view_id'].shape[1]
        for mod_name, mod_flags in zip(['rgb', 'ray', 'depth'], mod_flags.split(',')):
            for view_i, mod_flag in enumerate(mod_flags):
                if view_i < int(num_view):
                    batch['mods_flags'][mod_name][:, view_i] = MOD_FLAG_TABLE[mod_flag]
                    # force set first-view pose flag as condition
                    if mod_name == 'ray' and view_i == 0:
                        batch['mods_flags'][mod_name][:, view_i] = MOD_FLAG_TABLE['c']
        return batch
    
    def update(self, key, indices, values):
        if key in self.data:
            self.data[key][:, indices] = values
    
    def __call__(self, key):
        return self.data[key] if key in self.data else None
     



def tensor_recursive_to(d: dict, func):
    if isinstance(d, (list)):
        iterator = range(len(d))
    elif isinstance(d, dict):
        iterator = d.keys()
    for it in iterator:
        if isinstance(d[it], (list, dict, tuple)):
            if isinstance(d[it], tuple):
                d[it] = list(d[it])
            tensor_recursive_to(d[it], func)
        elif isinstance(d[it], (int, float, str, np.ndarray, PerspectiveCameras)):
            pass
        elif d[it] == None:
            pass
        else:
            d[it] = func(d[it])
            
            
def save_compare_image(np_image, path):
    N, H, W, C = np_image.shape
    np_image = np_image.transpose(1, 0, 2, 3).reshape(H, N * W, C)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np_image[..., ::-1])


def get_rgbd_point_cloud_numpy(cam, images, depths, depth_masks=None, mask_thr=None):
    point_cloud = get_rgbd_point_cloud(cam, images, depths, mask=depth_masks, mask_thr=mask_thr)
    points, colors = point_cloud.points_list()[0].detach().numpy(), point_cloud.features_list()[0].detach().numpy()
    # remove invalid points
    valid_mask = np.isfinite(points).all(axis=1)
    points, colors = points[valid_mask], colors[valid_mask]
    
    return points, colors