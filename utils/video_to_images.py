# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processes a video to a nerfstudio compatible dataset."""

import argparse
import os
import pathlib

from nerfstudio.process_data import process_data_utils
from nerfstudio.utils.rich_utils import CONSOLE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=pathlib.Path, default='x.mov')
    parser.add_argument('--output_folder', type=pathlib.Path, default='~/temp/')
    parser.add_argument('--num_frames_target', type=int, default=300)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--random_seed', type=int, default=1)

    args = parser.parse_args()
    data = args.data
    num_frames = args.num_frames_target
    output_folder = args.output_folder
    # scene_id = args.render_root.split('/')[-1]

    summary_log = []
    summary_log_eval = []
    # Convert video to images
    # create temp images folder to store the equirect and perspective images

    output_folder.mkdir(parents=True, exist_ok=True)
    summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
        data,
        image_dir=output_folder,
        num_frames_target=num_frames,
        num_downscales=0,
        crop_factor=(0.0, 0.0, 0.0, 0.0),
        verbose=args.verbose,
        random_seed=args.random_seed
    )

    CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

    for summary in summary_log:
        CONSOLE.log(summary)

