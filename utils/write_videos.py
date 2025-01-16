#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse
import os
import cv2
import imageio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_root', type=str, default='logs/exp-xxxxx/renders/xxxxx-train-set')
    parser.add_argument('--num_samples', type=int, default=80)
    parser.add_argument('--num_splines', type=int, default=3)
    parser.add_argument('--type', type=str, default='scene')
    
    args = parser.parse_args()
    render_folder = os.path.join(args.render_root, 'train')
    pred_root = os.path.join(render_folder, 'rgb')
    num_frames = args.num_samples
    output_folder = os.path.dirname(os.path.dirname(args.render_root))
    # scene_id = args.render_root.split('/')[-1]
    
    if args.type == 'scene':
        all_frames = sorted(os.listdir(pred_root))
        for i in range(args.num_splines):
            st_id, ed_id = i * num_frames, (i + 1) * num_frames
            img_list = []
            for j in range(st_id, ed_id):
                file = os.path.join(pred_root, f'frame_{j:04d}.png')
                img_list.append(cv2.imread(file)[..., ::-1])
            video_file = os.path.join(output_folder, f'3DGS-render-traj{i}.mp4')
            imageio.mimsave(video_file, img_list, fps=30)
    elif args.type == 'object':
        all_frames = sorted(os.listdir(pred_root))     
        for i in range(1):
            st_id, ed_id = i * num_frames, (i + 1) * num_frames
            img_list = []
            first_view_file = os.path.join(pred_root, 'ref_frame_0000.png')
            img_list.append(cv2.imread(first_view_file)[..., ::-1])
            for j in range(st_id, ed_id - 1):
                file = os.path.join(pred_root, f'frame_{j:04d}.png')
                img_list.append(cv2.imread(file)[..., ::-1])
            video_file = os.path.join(output_folder, f'3DGS-render-traj.mp4')
            imageio.mimsave(video_file, img_list, fps=30)            