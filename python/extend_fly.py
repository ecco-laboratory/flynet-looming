# %%
# imports

import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from video_dtw_utils import read_and_calc_video_flow
from flynet_utils import FlyNet, MegaFlyNet, get_disk_mask

# %%
# Paths and shit

# Need to set repo path because I suspect
# when slurm runs this it doesn't immediately know what is up
repo_path = '/home/mthieu/Repos/emonet-py/'
model_path = os.path.join(repo_path, 'ignore', 'models')
output_path = os.path.join(repo_path, 'ignore', 'outputs')
local_ck_path = '/home/mthieu/Repos/CowenKeltner'
video_path = os.path.join(local_ck_path, 'videos_10fps')
metadata_path = os.path.join(local_ck_path, 'metadata')



# %%
# Is MegaFlyNet the way...?

n_units_zhou2022 = 256

# 228x228 px seems to work ok with this
megaflynet = MegaFlyNet(conv_stride=8)

disk_mask_4ch = np.stack([get_disk_mask(12,4)] * 4, axis=-1)

conv_filter = np.load(os.path.join(model_path, 'zhou2022_{}unit_weights.npy'.format(n_units_zhou2022)))
conv_filter = conv_filter.reshape((12, 12, 4))
# masking behaves weird on tensors... leave it as np until the end
conv_filter[disk_mask_4ch] = 0.
conv_filter = torch.tensor(conv_filter)
conv_bias = torch.tensor(np.load(os.path.join(model_path, 'zhou2022_{}unit_intercept.npy'.format(n_units_zhou2022))))
classifier_weight = torch.tensor(np.load(os.path.join(model_path, 'zhou2022_{}unit_classifier_a.npy'.format(n_units_zhou2022))), dtype=torch.float32)
classifier_bias = torch.tensor(np.load(os.path.join(model_path, 'zhou2022_{}unit_classifier_b.npy'.format(n_units_zhou2022))))

megaflynet.conv.weight.data = conv_filter.moveaxis(-1, 0).unsqueeze(0)
megaflynet.conv.bias.data = conv_bias
# apparently, values of 1 get read in as 0-D scalars. weird
# needs to be double unsqueezed to get to a 1x1 matrix
megaflynet.classifier.weight.data = classifier_weight.unsqueeze(0).unsqueeze(0)
megaflynet.classifier.bias.data = classifier_bias

# Frozen! No training!
for param in megaflynet.parameters():
    param.requires_grad = False

# %%
# Reading flow directly in from videos cause... I don't give a fuck rn

video_ids = []
frame_ids = []
flynet_hit_probs = []
mean_flow_x = []
mean_flow_y = []

for in_folder, pwd, files in os.walk(video_path):
    # get the subfolder name
    in_subfolder = in_folder.split('/')[-1]

    # for C&K spoopy videos: change to if in_subfolder NOT in 5-class list
    if in_subfolder  == 'videos_10fps':
        continue
    
    for in_name in tqdm(files, desc=in_subfolder, leave=False):
        if in_name.endswith('.mp4'):
            in_file = os.path.join(in_folder, in_name)

            # Actually calculate stuff
            flow = read_and_calc_video_flow(in_file, resize=(228,228))
            flow_4d = convert_flow_numpy_to_tensor(flow)

            for frame in range(flow_4d.size()[0]):
                hit_prob = megaflynet(flow_4d[frame, ...].unsqueeze(0))
                video_ids.append(in_name)
                frame_ids.append(frame)
                flynet_hit_probs.append(hit_prob.numpy())
                mean_flow_x.append(flow[frame, :, :, 0].mean())
                mean_flow_y.append(flow[frame, :, :, 1].mean())

flynet_hit_probs = np.concatenate(flynet_hit_probs)

hit_prob_df = pd.DataFrame({
    'video': video_ids,
    'frame': frame_ids,
    'hit_prob': flynet_hit_probs,
    'mean_flow_x': mean_flow_x,
    'mean_flow_y': mean_flow_y
})

hit_prob_df = hit_prob_df.set_index(['video', 'frame'])
hit_prob_df.to_csv(os.path.join(metadata_path, 'flynet_hit_probs.csv'))

# %%
# Extract FlyNet "RF" activations

activations_all = []

for in_folder, pwd, files in os.walk(video_path):
    # get the subfolder name
    in_subfolder = in_folder.split('/')[-1]

    # for C&K spoopy videos: change to if in_subfolder NOT in 5-class list
    if in_subfolder  == 'videos_10fps':
        continue
    
    for in_name in tqdm(files, desc=in_subfolder, leave=False):
        if in_name.endswith('.mp4'):
            in_file = os.path.join(in_folder, in_name)

            # Actually calculate stuff
            flow = read_and_calc_video_flow(in_file, resize=(228,228))
            flow_4d = convert_flow_numpy_to_tensor(flow)

            # Pass through the conv layer
            activations = megaflynet.conv(flow_4d)
            # Keep the frame dimension, flatten the RF-pix dimensions
            activations = activations.flatten(start_dim=1)
            # I find it easiest to add the frame info on now as a df index
            activations = pd.DataFrame(
                activations.numpy(),
                index = range(activations.size()[0])
            )
            activations.index.name = 'frame'
            # Manually add the video name on as an index now as well
            activations['video'] = in_name
            activations.set_index('video', append=True)

            activations_all.append(activations)

activations_all = pd.concat(activations_all)

activations_all.to_csv(os.path.join(metadata_path, 'flynet_228x228_stride8_activations.csv'))

# %%
# Figuring out how to read in the images

# Read in image
# Convert to numpy array
# Index only the R and B dimensions
# Convert to np.float32 before doing ANY maths because int math is crazy
# Re-center (subtract 128)
