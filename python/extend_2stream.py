# %%
# important imports

import os
import platform

import numpy as np
import pandas as pd
import torch
from wang2015_2stream_vgg16 import flow_vgg16, rgb_vgg16
from ucf101_utils import UCF101TwoStreamDataset
from emonet_utils import (Cowen2017Dataset,
                          emonet_output_classes, get_target_emotion_index)

from torchvision import transforms
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

model_path = '../ignore/models'
local_ck_path = '/home/mthieu/Repos/CowenKeltner'
local_ucf101_path = '/home/data/shared/UCF-101'
local_2stream_path = '/home/mthieu/Repos/two-stream-pytorch'

# %%
# Initialize the streams (they don't cross yet)
# 101 classes in UCF101 (hence the name)
dorsal_stream = flow_vgg16(pretrained=True, num_classes=101)
ventral_stream = rgb_vgg16(pretrained=True, num_classes=101)

# Apparently, according to Simonyan & Zisserman, the streams only cross at the VERY end
# Like, literally, you average the class probabilities. Fine then lol

# %%
# Load the data
annotation_path = os.path.join(local_2stream_path, 'datasets/settings/ucf101')
these_transforms = transforms.Compose([
    transforms.Resize((256)), # smallest edge to 256
    transforms.RandomResizedCrop(224, (0.4, 0.8)),
    transforms.RandomHorizontalFlip()
])

twostream_split1_train = UCF101TwoStreamDataset(
    root=os.path.join(local_ucf101_path),
    annPath=annotation_path,
    train=True,
    split=1,
    transform=these_transforms
)

# %%
# 