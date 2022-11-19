# %%
# imports

import os

import numpy as np
import torch
from torch import nn

model_path = '../ignore/models'
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
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=4,
            out_channels=1,
            kernel_size=12,
            stride=11
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
        # THEN!! the comment says average but the code says sum
        # either way, reduce across timepoints
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
# %%
# Testing to make sure nn.Linear is equivalent to Baohua's channel-wise tensordot

# numpy array, mind you!
disk_mask_flat = get_disk_mask(12,4).flatten()
disk_mask_4ch = np.stack([disk_mask_flat] * 4, axis=1)

weights32 = np.load('../ignore/models/zhou2022_32unit_weights.npy')
weights32[disk_mask_4ch] = 0.
weights32_tensor = torch.Tensor(weights32)

weights_small = torch.randn((9, 2))

test_flatten = nn.Flatten(-2, -1)
test_linear = nn.Linear(18, 1, bias=False)
test_linear.weight.data = weights_small.flatten().unsqueeze(0)

# fake: batch size 3, 10 timepoints, 32 units, 144 RFpixels, 4 flowchannels
# x = torch.randn((3, 10, 32, 144, 4))
# super minimal test that I can hopefully manipulate by hand:
# 4 units, 9 pixels, 2 channels
x = torch.randn((4, 9, 2))
result_nn = test_linear(test_flatten(x)).detach().squeeze()
# Mathematically expected to be identical to nn.Linear
result_tensordot_flatten = torch.tensordot(torch.flatten(x, -2, -1), weights_small.flatten(), dims=[[-1], [0]])
# Returning sliiiightly different for some reason...
result_tensordot = torch.sum(torch.stack([torch.tensordot(x[..., i], weights_small[..., i], dims=[[-1],[0]]) for i in range(x.size()[-1])], dim=-1), dim=-1)
# %%
# Instantiate a flynet, load in Baohua's weights

flynet = FlyNet()

linear_weight = torch.Tensor(np.load(os.path.join(model_path, 'zhou2022_32unit_weights.npy')))
linear_bias = torch.Tensor(np.load(os.path.join(model_path, 'zhou2022_32unit_intercept.npy')))
classifier_weight = torch.Tensor(np.load(os.path.join(model_path, 'zhou2022_32unit_classifier_a.npy')))
classifier_bias = torch.Tensor(np.load(os.path.join(model_path, 'zhou2022_32unit_classifier_b.npy')))

# numpy array, mind you!
disk_mask_flat = get_disk_mask(12,4).flatten()
disk_mask_4ch = np.stack([disk_mask_flat] * 4, axis=1)
# Go ahead and mask out the weights outside the circular RF
linear_weight[disk_mask_4ch] = 0.

flynet.linear.weight.data = linear_weight.flatten().unsqueeze(0)
flynet.linear.bias.data = linear_bias
flynet.classifier.weight.data = classifier_weight.unsqueeze(0)
flynet.classifier.bias.data = classifier_bias

# Frozen! No training!
for param in flynet.parameters():
    param.requires_grad = False
# %%
# Is MegaFlyNet the way...?

megaflynet = MegaFlyNet()

disk_mask_4ch = np.stack([get_disk_mask(12,4)] * 4, axis=-1)

conv_filter = np.load(os.path.join(model_path, 'zhou2022_256unit_weights.npy'))
conv_filter = conv_filter.reshape((12, 12, 4))
# masking behaves weird on tensors... leave it as np until the end
conv_filter[disk_mask_4ch] = 0.
conv_filter = torch.tensor(conv_filter)
conv_bias = torch.tensor(np.load(os.path.join(model_path, 'zhou2022_256unit_intercept.npy')))
classifier_weight = torch.tensor(np.load(os.path.join(model_path, 'zhou2022_256unit_classifier_a.npy')), dtype=torch.float32)
classifier_bias = torch.tensor(np.load(os.path.join(model_path, 'zhou2022_256unit_classifier_b.npy')))

megaflynet.conv.weight.data = conv_filter.moveaxis(-1, 0).unsqueeze(0)
megaflynet.conv.bias.data = conv_bias
# apparently, values of 1 get read in as 0-D scalars. weird
# needs to be double unsqueezed to get to a 1x1 matrix
megaflynet.classifier.weight.data = classifier_weight.unsqueeze(0).unsqueeze(0)
megaflynet.classifier.bias.data = classifier_bias

# Frozen! No training!
for param in megaflynet.parameters():
    param.requires_grad = False

# %%
# Figuring out how to read in the images

# Read in image
# Convert to numpy array
# Index only the R and B dimensions
# Convert to np.float32 before doing ANY maths because int math is crazy
# Re-center (subtract 128)