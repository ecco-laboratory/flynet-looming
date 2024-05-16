# %%
# imports

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch

from myutils.flynet_utils import MegaFlyNet, get_disk_mask

# %%
# Paths and shit

# Need to set repo path because I suspect
# when slurm runs this it doesn't immediately know what is up
repo_path = '/home/mthieu/Repos/emonet-py/'
model_path = os.path.join(repo_path, 'ignore', 'models')
zhou2022_path = os.path.join(model_path, 'zhou2022')

# %%
# Arg definition and capture
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-u',
    '--units',
    default=256,
    type=int,
    choices=[32,256],
    help="Which model (32 or 256 units) to pull pre-trained kernel filter from?"
)
args = vars(parser.parse_args())

n_units_zhou2022 = args['units']

# %%
# Dewit

# Load in Baohua's weights
conv_filter = np.load(os.path.join(zhou2022_path, 'zhou2022_{}unit_weights.npy'.format(n_units_zhou2022)))
conv_bias = torch.tensor(np.load(os.path.join(zhou2022_path, 'zhou2022_{}unit_intercept.npy'.format(n_units_zhou2022))))
classifier_weight = torch.tensor(np.load(os.path.join(zhou2022_path, 'zhou2022_{}unit_classifier_a.npy'.format(n_units_zhou2022))), dtype=torch.float32)
classifier_bias = torch.tensor(np.load(os.path.join(zhou2022_path, 'zhou2022_{}unit_classifier_b.npy'.format(n_units_zhou2022))))

# Mask out corner pixels so that the conv filter is circular
# masking behaves weird on tensors... leave it as np until the end
disk_mask_4ch = np.stack([get_disk_mask(12,4)] * 4, axis=-1)
conv_filter = conv_filter.reshape((12, 12, 4))
conv_filter[disk_mask_4ch] = 0.
conv_filter = torch.tensor(conv_filter)

# Slap those weights into an instance of PyTorch-style MegaFlyNet
megaflynet = MegaFlyNet()
megaflynet.conv.weight.data = conv_filter.moveaxis(-1, 0).unsqueeze(0)
megaflynet.conv.bias.data = conv_bias
# apparently, values of 1 get read in as 0-D scalars. weird
# needs to be double unsqueezed to get to a 1x1 matrix
megaflynet.classifier.weight.data = classifier_weight.unsqueeze(0).unsqueeze(0)
megaflynet.classifier.bias.data = classifier_bias

# Save the weights out as a .pt file so I don't have to run this every time
torch.save(megaflynet.state_dict(), os.path.join(model_path, 'MegaFlyNet{}.pt'.format(n_units_zhou2022)))
