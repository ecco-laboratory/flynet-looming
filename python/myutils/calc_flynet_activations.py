# %%
# imports

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from video_dtw_utils import read_and_calc_video_flow
from flynet_utils import MegaFlyNet, convert_flow_numpy_to_tensor

# %%
# Argle parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-l',
    '--length',
    default=132,
    type=int,
    help="Image length/height/width in px to resize to (imgs are square)"
)
parser.add_argument(
    '-i',
    '--in_paths',
    nargs='*',
    type=str,
    help="Paths to videos. Input multiple paths for multiple videos"
)
parser.add_argument(
    '-o',
    '--out_path',
    type=str,
    help="Path to write out overall activations file for all videos input"
)
parser.add_argument(
    '-w',
    '--weight_path',
    type=str,
    help="Which model path to pull pre-trained kernel filter from?"
)
parser.add_argument(
    '-q',
    '--quantity_to_calc',
    type=str,
    choices=['hit_probs', 'activations'],
    help="Which FlyNet quantity to calculate? Hit probs or kernel activations?"
)
parser.add_argument(
    '-s',
    '--stride',
    default=8,
    type=int,
    help="FlyNet kernel stride length (passed onto PyTorch)"
)

args = vars(parser.parse_args())

imgsize = args['length']
stride = args['stride']
weight_path = args['weight_path']
quantity_to_calc = args['quantity_to_calc']
video_paths = args['in_paths']
out_path = args['out_path']

# %%
# Init MegaFlyNet instance
# stride = 8 seems like an acceptable amount of overlap between kernel steps
megaflynet = MegaFlyNet(conv_stride=stride)
megaflynet.load_state_dict(torch.load(weight_path))

# Frozen! No training!
for param in megaflynet.parameters():
    param.requires_grad = False

# %%
# helper function definition
def calc_flynet_hit_prob(flow, convmodel):
    # Initialize var for output info
    # will be bound into df later, right now easiest to append with a list
    hit_probs = []

    # Frame-by-frame hit probabilities
    # It's wrapped in a for loop so we can call megaflynet separately on every frame
    # because normally if you call it on a video it sums over frames
    for frame in range(flow.size()[0]):
        hit_prob = convmodel(flow[frame, ...].unsqueeze(0))
        hit_probs.append(hit_prob.numpy())
    
    hit_probs_df = pd.DataFrame({
        'frame': range(len(hit_probs)),
        'hit_prob': np.concatenate(hit_probs),
    })
    hit_probs_df = hit_probs_df.set_index('frame')

    return hit_probs_df

def calc_flynet_activation(flow, convmodel):
    # Frame by frame RF 'activations'
    # Don't need to pass through a for loop because the component functions keep stuff separate by frame
    # Pass through the conv layer
    activations = convmodel.conv(flow)
    # Keep the frame dimension, flatten the RF-pix dimensions
    activations = activations.flatten(start_dim=1)
    # I find it easiest to add the frame info on now as a df index
    # note that the colnames of the activations will come out just as the digit indices of the units
    activations = pd.DataFrame(
        activations.numpy(),
        index = range(activations.size()[0])
    )
    activations.index.name = 'frame'

    return activations

# %%
# Workhorse

# Reading flow directly in from videos cause... I don't give a fuck rn
# But actually it's better to calculate flow fresh in case we change the img size

out_all = []
    
for in_file in tqdm(video_paths, desc='Stimulus videos'):

    # Get that fresh flow
    flow = read_and_calc_video_flow(in_file, resize=(imgsize,imgsize))
    flow = convert_flow_numpy_to_tensor(flow)

    # Actually calculate stuff
    if quantity_to_calc == 'hit_probs':
        out = calc_flynet_hit_prob(flow, megaflynet)
    elif quantity_to_calc == 'activations':
        out = calc_flynet_activation(flow, megaflynet)            

    # Manually add the video name on as an index now as well
    out['video'] = os.path.split(in_file)[1]
    out = out.set_index('video', append=True)

    # Now bind onto the superlists
    out_all.append(out)

out_all = pd.concat(out_all)
# old file naming convention that was set in this script
# that other targets may be expecting
# 'flynet_{}x{}_stride{}_{}.csv'.format(imgsize, imgsize, stride, quantity_to_calc)
out_all.to_csv(out_path)
