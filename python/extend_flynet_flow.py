# %%
# imports

import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from myutils.video_dtw_utils import read_and_calc_video_flow
from myutils.flynet_utils import MegaFlyNet, convert_flow_numpy_to_tensor
from tqdm import trange

# %%
# set useful paths or something
repo_path = '/home/mthieu/Repos/emonet-py/'
stimuli_path = '/home/mthieu/stimuli/'
output_path = os.path.join(repo_path, 'ignore', 'outputs')

# %%
# Estimate dense optical flow for the movie Split (James McAvoy being creppy)
movie_path = '/home/data/eccolab/OpenNeuro/ds002837/stimuli/split.mp4'
# The movie is 25 fps
flow = read_and_calc_video_flow(movie_path, resize=(228,228), progress=True)

# %%
# Estimate dense optical flow for StudyForrest extension retinotopy stimuli
studyforrest_retinotopy_stimuli_path = os.path.join(stimuli_path, 'studyforrest_retinotopy')
flow_ring_expand = read_and_calc_video_flow(os.path.join(studyforrest_retinotopy_stimuli_path, 'ring_expand.mp4'), resize=(132,132), progress=True)
flow_ring_contract = read_and_calc_video_flow(os.path.join(studyforrest_retinotopy_stimuli_path, 'ring_contract.mp4'), resize=(132,132), progress=True)
flow_wedge_clock = read_and_calc_video_flow(os.path.join(studyforrest_retinotopy_stimuli_path, 'wedge_clock.mp4'), resize=(132,132), progress=True)
flow_wedge_counter = read_and_calc_video_flow(os.path.join(studyforrest_retinotopy_stimuli_path, 'wedge_counter.mp4'), resize=(132,132), progress=True)
# %%
# Estimate dense optical flow for NSD retinotopy stimuli
nsd_retinotopy_stimuli_path = os.path.join(stimuli_path, 'nsd_retinotopy')
flow_nsd_bar = read_and_calc_video_flow(os.path.join(nsd_retinotopy_stimuli_path, 'RETBAR.mp4'), resize=(132,132), progress=True)
flow_nsd_wedgering = read_and_calc_video_flow(os.path.join(nsd_retinotopy_stimuli_path, 'RETWEDGERINGMASH.mp4'), resize=(132,132), progress=True)
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
# estimate studyforrest retinotopy "looming"
tensor_ring_expand = convert_flow_numpy_to_tensor(flow_ring_expand)
tensor_ring_contract = convert_flow_numpy_to_tensor(flow_ring_contract)
tensor_wedge_clock = convert_flow_numpy_to_tensor(flow_wedge_clock)
tensor_wedge_counter = convert_flow_numpy_to_tensor(flow_wedge_counter)

hit_probs_ring_expand = np.stack([megaflynet(tensor_ring_expand[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_ring_expand.size()[0])], axis=0)
hit_probs_ring_contract = np.stack([megaflynet(tensor_ring_contract[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_ring_contract.size()[0])], axis=0)
hit_probs_wedge_clock = np.stack([megaflynet(tensor_wedge_clock[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_wedge_clock.size()[0])], axis=0)
hit_probs_wedge_counter = np.stack([megaflynet(tensor_wedge_counter[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_wedge_counter.size()[0])], axis=0)

activations_ring_expand = megaflynet.conv(tensor_ring_expand).reshape((tensor_ring_expand.size()[0], -1)).numpy()
activations_ring_contract = megaflynet.conv(tensor_ring_contract).reshape((tensor_ring_contract.size()[0], -1)).numpy()
activations_wedge_clock = megaflynet.conv(tensor_wedge_clock).reshape((tensor_wedge_clock.size()[0], -1)).numpy()
activations_wedge_counter = megaflynet.conv(tensor_wedge_counter).reshape((tensor_wedge_counter.size()[0], -1)).numpy()

# %%
# estimate NSD retinotopy "looming"
tensor_nsd_bar = convert_flow_numpy_to_tensor(flow_nsd_bar)
tensor_nsd_wedgering = convert_flow_numpy_to_tensor(flow_nsd_wedgering)

hit_probs_nsd_bar = np.stack([megaflynet(tensor_nsd_bar[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_nsd_bar.size()[0])], axis=0)
hit_probs_nsd_wedgering = np.stack([megaflynet(tensor_nsd_wedgering[frame, ...].unsqueeze(0)).numpy() for frame in range(tensor_nsd_wedgering.size()[0])], axis=0)

activations_nsd_bar = megaflynet.conv(tensor_nsd_bar).reshape((tensor_nsd_bar.size()[0], -1)).numpy()
activations_nsd_wedgering = megaflynet.conv(tensor_nsd_wedgering).reshape((tensor_nsd_wedgering.size()[0], -1)).numpy()

# %%
# write studyforrest retinotopy flynet activations to file
np.savetxt(
    os.path.join(output_path, 'flynet_132x132_stride8_activations_studyforrest_retinotopy_ring_expand.csv'),
    activations_ring_expand,
    fmt='%1.6f',
    delimiter=','
    )
np.savetxt(
    os.path.join(output_path, 'flynet_132x132_stride8_activations_studyforrest_retinotopy_ring_contract.csv'),
    activations_ring_contract,
    fmt='%1.6f',
    delimiter=','
)
np.savetxt(
    os.path.join(output_path, 'flynet_132x132_stride8_activations_studyforrest_retinotopy_wedge_clock.csv'),
    activations_wedge_clock,
    fmt='%1.6f',
    delimiter=','
)
np.savetxt(
    os.path.join(output_path, 'flynet_132x132_stride8_activations_studyforrest_retinotopy_wedge_counter.csv'),
    activations_wedge_counter,
    fmt='%1.6f',
    delimiter=','
)
# %%
# write nsd retinotopy flynet activations to file
np.savetxt(
    os.path.join(output_path, 'flynet_132x132_stride8_activations_nsd_retinotopy_bar.csv'),
    activations_nsd_bar,
    fmt='%1.6f',
    delimiter=','
)
np.savetxt(
    os.path.join(output_path, 'flynet_132x132_stride8_activations_nsd_retinotopy_wedgering.csv'),
    activations_nsd_wedgering,
    fmt='%1.6f',
    delimiter=','
)

# %%
# Visualize studyforrest retinotopy hit probs
plt.figure(figsize=(9,1))
# display(plt.plot(hit_probs_ring_expand[0:800]))
display(plt.plot(hit_probs_ring_contract[0:800]))

# %%
plt.figure(figsize=(9,1))
display(plt.plot(hit_probs_wedge_clock[0:800]))
display(plt.plot(hit_probs_wedge_counter[0:800]))
# %%
