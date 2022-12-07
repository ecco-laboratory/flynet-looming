# %%
# importz

import os
# import argparse

# sys.path.append('../')

from video_dtw_utils import (convert_flow_to_rgb_cart, read_and_calc_video_flow, read_and_resize_video_frames, write_arrays_to_imgs)
from tqdm import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument('--in_file', help='absolute path to video file')
# take only the out root and not the changing out folder
# because it's harder for me to cycle the out folders in parallel in bash
# parser.add_argument('--out_root', help='absolute path to output ROOT folder for flow imgs')
# args = parser.parse_args()

# %%
# Write out the flow frames for a SINGLE video

out_root = '/home/mthieu/Repos/CowenKeltner/flowframes_10fps'

for in_folder, pwd, files in os.walk('/home/mthieu/Repos/CowenKeltner/videos_10fps'):
    # get the subfolder name
    in_subfolder = in_folder.split('/')[-1]

    # for C&K spoopy videos: change to if in_subfolder NOT in 5-class list
    if in_subfolder in ['videos_10fps', 'Anxiety', 'Excitement', 'Fear', 'Horror', 'Surprise']:
        continue

    # Construct the full valid out_folder path from the root and the in folder
    out_folder = os.path.join(out_root, in_subfolder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    for in_name in tqdm(files, desc=in_subfolder, leave=False):
        if in_name.endswith('.mp4'):
            in_file = os.path.join(in_folder, in_name)

            # Substring to drop the file extension
            out_file_stem = os.path.join(out_folder, in_name)[:-4]

            # Actually calculate stuff
            flow = read_and_calc_video_flow(in_file)
            flow_rgb = convert_flow_to_rgb_cart(flow)
            # frames = read_and_resize_video_frames(in_file)
            write_arrays_to_imgs(frames=flow_rgb, filename_stem=out_file_stem)

# %%
