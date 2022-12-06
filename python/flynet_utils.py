# %%
# imports

import numpy as np
import torch
from torch import nn

# %%
# Instantiate class for translated Zhou et al 2022 fly ANN

# for ONE unit:
# the first dimension is (flattened) pixel weights
# comes in as 144, aka 12x12
# the second dimension is cardinal direction: R, L, U, D in that order
# If the RF for each unit is a circle, the corner weights should be set to 0

# Each unit appears to have the same weights
# just with a different RF center

# From Baohua's code on GitHub:
# UV_flow_t: tensor, flow field at time point t, shape = batch_size by M by K*K by 4
# So remember: We are NOT operating in actual-pixel space, we are operating in RF space
# so the k-by-k'th pixel weight in unit m's RF does not correspond
# to the same actual-pixel as the k-by-k'th pixel weight in unit m+1's RF

# tensordot of a 3D flow tensor (indexing a single flow direction) with the weights (144x1)
# seems to yield an output size batch x m x 1
# making me think that a linear layer WILL do the same thing

# The model is ReLU(weighted sum of directional weights*flow values)
# Flatten? and then linear to get weighted sum? and then ReLU the linear output?
# Baohua's code appears to 

class MegaFlyNet(nn.Module):
    def __init__(self, conv_stride: int = 11) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=4,
            out_channels=1,
            kernel_size=12,
            stride=conv_stride
        )
        self.classifier = nn.Linear(1, 1)
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        # Should come out of the convolution as an 'image'
        # with M RFpixels, aka sqrt(M) x sqrt(M) size
        # and ONE channel
        # Batch size and time dimensions might come ahead of height and width
        # that's all fine by moi
        x = nn.functional.relu(x)
        # THEN sum across units within timepoint
        # this should leave a batch size x time x 1 array
        x = torch.sum(x, axis=[-2, -1])
        x = self.classifier(x)
        x = torch.sigmoid(x)
        # THEN!! average P(hit) across timepoints
        # The paper has stuff to do a weighted average to favor some timepoints
        # but I don't have the weights so, shrug
        # axis -2 because the last dim is still 1?
        x = torch.mean(x, axis=-2)
        # and then we out!
        # God I hope this is equivalent
        return x

class FlyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 144 x 4 input features
        # must be FLATTENED before feeding in to linear
        # Remember also to flatten Baohua's weights
        # BEFORE loading them into the layer weights attribute
        self.linear = nn.Linear(144*4, 1)
        self.classifier = nn.Linear(1, 1)
    
    def forward(self, x: torch.Tensor):
        x = self.linear(torch.flatten(x, -2, -1))
        # It should come out of that linear layer
        # with shape batch x time x unit x 1
        # THEN ReLU within batch x time x unit
        x = nn.functional.relu(x)
        # THEN sum across units within timepoint
        # this should leave a batch size x time x 1 array
        x = torch.sum(x, axis=-2)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        # THEN!! the comment says average but the code says sum
        # either way, reduce across timepoints
        x = torch.sum(x, axis=-2)
        # and then we out!
        return x

# %%
# Helper functions from Baohua's little vis notebook
# Delete these when you've figured out the RF corner zero issue
# Get the element center
def get_element_center(leftup_corner, L):
    """
    Args:
    leftup_corner: tuple, indices of the left-up corner of the element under consideration
    L: element dimension
    
    Return:
    element_center: indices of the element center
    """
    L_half = (L - 1) / 2.
    element_center = (leftup_corner[0] + L_half, leftup_corner[1] + L_half)
    
    return element_center


# Check whether within receptive field
def within_receptive(leftup_corner, K, L, pad):
    """
    Args:
    leftup_corner: tuple, indices of the left-up corner of the element under consideration
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    
    Return:
    within_resep: whether the element indicated by the leftup corner is within the receptive field. True or False.
    """
    N = K * L + 2 * pad
    N_half = (N - 1) / 2.
    element_center = get_element_center(leftup_corner, L)
    d = np.sqrt((element_center[0]-N_half)**2 + (element_center[1]-N_half)**2)
    within_resep = d <= N_half - pad
    
    return within_resep


# Get the indices of the left-up corner of each element given the dimension of the frame N and # of element centers K*K
def get_leftup_corners(K, L, pad):
    """
    Args:
    K: K*K is the totoal # of elements
    L: dimension of each element
    pad: padding size
    
    Returns:
    leftup_corners: indices of the left-up corner of each element on the frame
    """
    leftup_corners = []
    for row in range(K):
        for col in range(K):
            row_value = row * L + pad
            col_value = col * L + pad
            leftup_corners.append([row_value, col_value])
            
    return np.array(leftup_corners)


# Get disk mask
def get_disk_mask(K, L):
    """
    Args:
    K: K*K is the total # of elements
    L: element dimension
    
    Returns:
    disk_mask: boolean, disk mask
    """
    disk_mask = np.full(K*K, True)
    leftup_corners = get_leftup_corners(K, L, 0)
    for counter, leftup_corner in enumerate(leftup_corners):
        if within_receptive(leftup_corner, K, L, 0):
            disk_mask[counter] = False
    
    return disk_mask.reshape((K, K))
