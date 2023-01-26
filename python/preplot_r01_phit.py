# %%
# imports

import os

import numpy as np
import pandas as pd
import torch
from myutils.video_dtw_utils import read_and_calc_video_flow
from myutils.flynet_utils import MegaFlyNet, convert_flow_numpy_to_tensor

# %%
# set useful paths or something
repo_path = '/home/mthieu/Repos/emonet-py/'
output_path = os.path.join(repo_path, 'ignore', 'outputs', 'r01_app')

# %%
# Load FlyNet
megaflynet = MegaFlyNet(conv_stride=8)
megaflynet.load_state_dict(torch.load('/home/mthieu/Repos/emonet-py/ignore/models/MegaFlyNet256.pt'))

# Frozen! No training!
for param in megaflynet.parameters():
    param.requires_grad = False

megaflynet.eval()

# Helper function to run FlyNet separately on each frame to get hit prob timeseries
def get_hit_prob_timeseries(frames):
    tensor_frames = convert_flow_numpy_to_tensor(frames)
    hit_probs = np.stack([megaflynet(tensor_frames[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_frames.size()[0])], axis=0)

    return hit_probs

# %%
# Estimate FlyNet hit probs for a variety of stimuli
flow_clery_2020 = read_and_calc_video_flow(
    '/home/mthieu/stimuli/clery_2020_marmoset_stimulus.mp4',
    resize=(132,132)
)
flow_lourenco = read_and_calc_video_flow(
    '/home/mthieu/stimuli/lourenco_looming_rabbit.wmv',
    resize=(132,132)
)
flow_baseball = read_and_calc_video_flow(
    '/home/mthieu/Downloads/baseball.mp4',
    resize=(132,132)
)

hit_probs_clery_2020 = get_hit_prob_timeseries(flow_clery_2020)
hit_probs_lourenco = get_hit_prob_timeseries(flow_lourenco)
hit_probs_baseball = get_hit_prob_timeseries(flow_baseball)
# %%
np.savetxt(
    fname=os.path.join(output_path, 'hit_probs_lourenco.txt'),
    X=hit_probs_lourenco,
    fmt='%.6f'
)
np.savetxt(
    fname=os.path.join(output_path, 'hit_probs_baseball.txt'),
    X=hit_probs_baseball,
    fmt='%.6f'
)
