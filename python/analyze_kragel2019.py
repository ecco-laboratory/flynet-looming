# %%
# imports up top, babies

import platform

import pandas as pd
import torch
from emonet_utils import EmoNet
from torch import nn

# %%
# Set abs paths based on which cluster node we're on
base_path = '/data/eccolab/'
if platform.node() != 'ecco':
    base_path = '/home'+base_path

emonet_path = '../ignore/models/EmoNet.pt'
metadata_path = '~/Repos/CowenKeltner/metadata/'

# %%
# Fire up the ol net
emonet_torch = EmoNet()
emonet_torch.load_state_dict(state_dict=torch.load(emonet_path))

# Turn off backward gradients for all runs of the model
# Right now we just be inferencing
for param in emonet_torch.parameters():
    param.requires_grad = False

# %%
# Read in the Cowen & Keltner metadata

ck_metadata = pd.read_csv(metadata_path+'video_ratings.csv')
ck_metadata = ck_metadata.set_index('Filename')

# Truly I wish this was in long form but Alan doesn't like tidy data does he
ck_censored = pd.read_csv(metadata_path+'censored_video_ids.csv')

# We don't need to see the censored ones! At least I personally don't
# I guess the model doesn't have feelings
ck_metadata = ck_metadata[~ck_metadata.index.isin(ck_censored['less.bad'])]
ck_metadata = ck_metadata[~ck_metadata.index.isin(ck_censored['very.bad'])]
# Just the emotion categories
ck_metadata = ck_metadata.iloc[:, range(34)]
# %%
