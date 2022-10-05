# %%
# imports
from io import BytesIO
from turtle import forward
from typing import Any, Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

# %%
# read_and_preprocess_image

def read_and_preprocess_image(path, cropbox = None):
    if path.startswith('http'):
        # use requests and io.BytesIO to pull from web
        img = Image.open(BytesIO(requests.get(path).content))
    else:
        img = Image.open(path)
    
    # expect pre-formatted pillow cropbox
    if cropbox is not None:
        img = img.crop(cropbox)
    # resize and then render as numpy array
    # but leave the image in 0-255 RGB space as that's also how Matlab imread returns imgs
    # TODO: soft-code the dimensions to get the height and width of the img from the model
    img = np.asarray(img.resize((227, 227)))
    # move the RGB axis to from last to first
    # TODO: Figure out why the matlab code was reversing all the axes, not just the last one
    img = np.moveaxis(img, -1, 0)

    return img

# %%
# cropbox_nsd_to_pillow
def cropbox_nsd_to_pillow(size, cropbox):
    # expect size to come in from the COCO metadata
    # expect cropbox to come in from the NSD stim dataframe

    if sum(cropbox[0:2]) == 0:
        upper = 0
        left = size[0] * cropbox[2]
        lower = size[1]
        right = size[0] - (size[0] * cropbox[3])
    else:
        upper = size[1] * cropbox[0]
        left = 0
        lower = size[1] - (size[1] * cropbox[1])
        right = size[0]
    
    return (left, upper, right, lower)
# %%
# pad_sequence() with batch_size fixed to True for feeding in as collate_fn to DataLoader
def pad_sequence_batch_first(
    sequences: Union[Tensor, List[Tensor]]
    ) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

# %%
# Resize transform but reaches in to pull out the frames from a read_video obj
class ResizeVideo(torchvision.transforms.Resize):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(size, interpolation, max_size, antialias)
    
    def forward(self, vid):
        """
        Args:
            vid (torchvision video from read_video): Video to be scaled.

        Returns:
            Video: torchvision video with frames rescaled.
        """
        frames = vid[0]
        frames = F.resize(vid[0], self.size, self.interpolation, self.max_size, self.antialias)
        return (frames, vid[1], vid[2])
