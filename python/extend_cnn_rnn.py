# %%
# imports up top, babies

import os
import platform

import torch
import torchvision
from emonet_utils import EmoNet, EmoNetHeadlessVideo
from image_utils import Cowen2017Dataset, ResizeVideo

# from tqdm import tqdm

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

model_path = '../ignore/models'
metadata_path = '/home/mthieu/Repos/CowenKeltner/metadata'
# %%
# Fire up the ol net
emonet_torch = EmoNet()
emonet_torch.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'EmoNet.pt')))

# %%
# Fire up a newer net
emonet_headless = EmoNetHeadlessVideo()
emonet_headless.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'EmoNetHeadless.pt')))

for param in emonet_headless.parameters():
    param.requires_grad = False
# %%
# Load ze data
ck_torchdata = Cowen2017Dataset(root=os.path.join(base_path, 'CowenKeltner/Videos_by_Category'),
                                annPath=metadata_path,
                                train=False,
                                transform=ResizeVideo((227, 227))
                                )

ck_torchloader = torch.utils.data.DataLoader(ck_torchdata, batch_size=7)
# %%
