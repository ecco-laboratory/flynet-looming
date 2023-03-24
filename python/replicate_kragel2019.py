# %%
# imports up top, babies

import os
import platform

import numpy as np
import pandas as pd
import torch
import torchvision
from myutils.emonet_utils import Cowen2017Dataset, EmoNetPythonic, EmoNetHeadlessVideo, emonet_output_classes
from tqdm import tqdm

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

model_path = '../ignore/models'
local_ck_path = '/home/mthieu/Repos/CowenKeltner'
metadata_path = os.path.join(local_ck_path, 'metadata')

# %%
# Fire up the full EmoNet
emonet_torch = EmoNetPythonic()
emonet_torch.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'EmoNetPythonic.pt')))

# Turn off backward gradients for all runs of the model
# Right now we just be inferencing
for param in emonet_torch.parameters():
    param.requires_grad = False

# %%
# Fire up the headless EmoNet (outputs the penultimate activations)
emonet_headless = EmoNetHeadlessVideo()
emonet_headless.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'EmoNetHeadless.pt')))

# Turn off backward gradients for all runs of the model
# Right now we just be inferencing
for param in emonet_headless.parameters():
    param.requires_grad = False

# %%
# read those motherfuckin videos in

test = torchvision.io.read_video(os.path.join(base_path, 'CowenKeltner/Videos_by_Category/Adoration/0035.mp4'), pts_unit='sec')

ck_torchdata_train = Cowen2017Dataset(
    root=os.path.join(local_ck_path, 'videos_10fps'),
    annPath=metadata_path,
    censor=False,
    train=True,
    transform=torchvision.transforms.Resize((227, 227))
)

ck_torchdata_test = Cowen2017Dataset(
    root=os.path.join(local_ck_path, 'videos_10fps'),
    annPath=metadata_path,
    censor=False,
    train=False,
    transform=torchvision.transforms.Resize((227, 227))
)

# Set batch_size here to 1 so it's just one video at a time
# BUT! Each video effectively acts as a batch of frames, as long as time is in the first dim
ck_torchloader_train = torch.utils.data.DataLoader(ck_torchdata_train, batch_size=1)
ck_torchloader_test = torch.utils.data.DataLoader(ck_torchdata_test, batch_size=1)
# %%
# Let's get predicting (full EmoNet)
emonet_torch.eval()

preds_all = {}

# For full EmoNet, only need to do it on the test data 
# because it's been trained on the training data (duh)
for vid, lab in tqdm(ck_torchloader_test):
    vid = vid.squeeze()
    pred = emonet_torch(vid)
    preds_all[lab['id'][0]] = pred.numpy()

# %%
# Let's get predicting (headless EmoNet)
# We do need to do this on the training and testing data
# so subsequent discriminant analysis models can be trained on the training activations
# and tested on the testing activations
emonet_headless.eval()

preds_all = {}

for vid, lab in tqdm(ck_torchloader_train):
    vid = vid.squeeze() # must squeeze to drop the 5th batch dimension
    pred = emonet_headless(vid)
    pred = pred.mean(dim = 0) # Collapse over frames
    preds_all[lab['id'][0]] = pred.numpy()

for vid, lab in tqdm(ck_torchloader_test):
    vid = vid.squeeze()
    pred = emonet_headless(vid)
    pred = pred.mean(dim = 0) # Collapse over frames
    preds_all[lab['id'][0]] = pred.numpy()

# the video-wise predictions are! decently sized dataframe!
# For the 
for id in preds_all.keys():
    # Gotta add an empty 0 dimension to make it 1x4096 so it becomes a 1 row dataframe with 4096 cols
    preds_all[id] = pd.DataFrame(np.expand_dims(preds_all[id], 0))

# For the full EmoNet predictions, set column names using columns=emonet_output_classes
# And don't drop the frame index if you are outputting framewise predictions :)
preds_all = pd.concat(preds_all, names=['video', 'frame']).reset_index(level='frame',drop=True)

preds_all.to_csv(os.path.join(metadata_path, 'kragel2019_videowise_activations_fc7.csv'))
# %%
