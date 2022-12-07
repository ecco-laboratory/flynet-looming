# %%
# imports

import os

import numpy as np
import pandas as pd
import torch
from video_dtw_utils import read_and_calc_video_flow
from flynet_utils import MegaFlyNet, convert_flow_numpy_to_tensor
from tqdm import trange

# %%
# Estimate dense optical flow for the movie Split (James McAvoy being creppy)
movie_path = '/home/data/eccolab/OpenNeuro/ds002837/stimuli/split.mp4'
# The movie is 25 fps
flow = read_and_calc_video_flow(movie_path, resize=(228,228), progress=True)

# %%
# Load FlyNet
megaflynet = MegaFlyNet(conv_stride=8)
megaflynet.load_state_dict(torch.load('/home/mthieu/Repos/emonet-py/ignore/models/MegaFlyNet256.pt'))

# Frozen! No training!
for param in megaflynet.parameters():
    param.requires_grad = False

megaflynet.eval()

# %%
# estimate looming
hit_probs = []
for tr_idx in trange(int(flow.shape[0] / 50)):
    # Every 50 frames, aka 2 seconds
    # estimate the p_hit of the last 50 frames
    # and put it out in wide form for sloping later
    tr_idx = tr_idx + 1
    frame_idx = tr_idx * 50

    these_frames = flow[range(frame_idx-50, frame_idx), ...]
    these_frames = convert_flow_numpy_to_tensor(these_frames)
    these_hit_probs = np.stack([megaflynet(these_frames[idx, ...].unsqueeze(0)).numpy() for idx in range(50)], axis=1)
    hit_probs.append(these_hit_probs)

hit_probs = np.stack(hit_probs, axis=0)
# There is an extra dim coming in somewhere /shrug
hit_probs = hit_probs.squeeze()
# %%
hit_probs_df = pd.DataFrame(hit_probs)
hit_probs_df.to_csv('/home/mthieu/Repos/emonet-py/ignore/outputs/flynet_hitprobs_nndb_splitmovie.csv')
# %%
