# %%
# imports

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from myutils.video_dtw_utils import read_and_calc_video_flow
from myutils.flynet_utils import MegaFlyNet, convert_flow_numpy_to_tensor

# %%
# Paths and shit

# Need to set repo path because I suspect
# when slurm runs this it doesn't immediately know what is up
repo_path = '/home/mthieu/Repos/emonet-py/'
model_path = os.path.join(repo_path, 'ignore', 'models')

# %%
# Argle parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-h',
    '--height',
    type=int,
    help="Image height in px to resize to (imgs are square, this is width too)"
)
parser.add_argument(
    '-p',
    '--path',
    type=str,
    help="Root folder for videos and shit"
)
parser.add_argument(
    '-v',
    '--videos',
    type=str,
    help="Subfolder for videos specifically"
)
parser.add_argument(
    '-m',
    '--metadata',
    type=str,
    help="Subfolder for metadata and output files"
)
parser.add_argument(
    '-q',
    '--quantity-to-calc',
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
parser.add_argument(
    '-u',
    '--units',
    default=256,
    type=int,
    choices=[32,256],
    help="Which model (32 or 256 units) to pull pre-trained kernel filter from?"
)
args = vars(parser.parse_args())

imgsize = args['height']
stride = args['stride']
n_units = args['units']
base_path = args['path']
quantity_to_calc = args['quantity-to-calc']
video_path = os.path.join(base_path, args['videos'])
metadata_path = os.path.join(base_path, args['metadata'])

# %%
# Init MegaFlyNet instance
# stride = 8 seems like an acceptable amount of overlap between kernel steps
megaflynet = MegaFlyNet(conv_stride=stride)
megaflynet.load_state_dict(torch.load(os.path.join(model_path, 'MegaFlyNet{}.pt'.format(n_units))))

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

for in_folder, pwd, files in os.walk(video_path):
    # get the subfolder name
    in_subfolder = in_folder.split('/')[-1]

    # fencepost condition to skip if it's not actually a proper subfolder
    # but the current folder
    if in_subfolder == args['videos']:
        continue
    
    for in_name in tqdm(files, desc=in_subfolder, leave=False):
        if in_name.endswith('.mp4'):
            in_file = os.path.join(in_folder, in_name)

            # Get that fresh flow
            flow = read_and_calc_video_flow(in_file, resize=(imgsize,imgsize))
            flow = convert_flow_numpy_to_tensor(flow)

            # Actually calculate stuff
            if quantity_to_calc == 'hit_probs':
                out = calc_flynet_hit_prob(flow, megaflynet)
            elif quantity_to_calc == 'activations':
                out = calc_flynet_activation(flow, megaflynet)            

            # Manually add the video name on as an index now as well
            out['video'] = in_name
            out = out.set_index('video', append=True)

            # Now bind onto the superlists
            out_all.append(out)

out_all = pd.concat(out_all)    
out_all.to_csv(os.path.join(metadata_path, 'flynet_{}x{}_stride{}_{}.csv'.format(imgsize, imgsize, stride, quantity_to_calc)))

# %%
# Figuring out how to read in the images

# Read in image
# Convert to numpy array
# Index only the R and B dimensions
# Convert to np.float32 before doing ANY maths because int math is crazy
# Re-center (subtract 128)
