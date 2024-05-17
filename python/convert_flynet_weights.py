# %%
# imports

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch

from myutils.flynet_utils import MegaFlyNet, get_disk_mask

# %%
# Arg definition and capture
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-i',
    '--in_folder',
    type=str,
    help="Folder where Zhou et al. (2022) pre-trained kernel filter weights are saved"
)
parser.add_argument(
    '-o',
    '--out_path',
    type=str,
    help="Full path and filename to write out PyTorch-ified weights"
)

args = vars(parser.parse_args())

zhou2022_path = args['in_folder']
model_path = args['out_path']

# %%
# Dewit

# Load in Baohua's weights
conv_filter = np.load(os.path.join(zhou2022_path, 'zhou2022_256unit_weights.npy'))
conv_bias = torch.tensor(np.load(os.path.join(zhou2022_path, 'zhou2022_256unit_intercept.npy')))
classifier_weight = torch.tensor(np.load(os.path.join(zhou2022_path, 'zhou2022_256unit_classifier_a.npy')), dtype=torch.float32)
classifier_bias = torch.tensor(np.load(os.path.join(zhou2022_path, 'zhou2022_256unit_classifier_b.npy')))

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
torch.save(megaflynet.state_dict(), model_path)
