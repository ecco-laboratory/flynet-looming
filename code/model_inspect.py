# %%
# imports
import platform

import pandas as pd
import torch
import torchvision
from image_utils import NSDDataset
from torch import nn
from torchvision import transforms
from tqdm import tqdm

# %%
# Set abs paths based on which cluster node we're on
base_path = '/data/eccolab/'
if platform.node() != 'ecco':
    base_path = '/home'+base_path

emonet_path = '../ignore/models/EmoNet.pt'
nsd_path = base_path+'Code/NSD/'
# %%
# initialize EmoNet class with correct param settings
alexnet_lrn_alpha = 9.999999747378752e-05
# Can't use torchvision.models.AlexNet()
# because it comes out slightly diff from Matlab. Lol
class EmoNet(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        # TODO: still needs the onnx initializer, whatever that thing is
        # Initialized with an empty module?
        self.initializers = nn.Module()
        # initializer has one weight per RGB per pixel
        self.initializers.register_parameter(name='onnx_initializer_0',param=nn.Parameter(torch.ones(torch.Size([1, 3, 227, 227]))))
        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.Conv_0 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # Adjust each weight independently
        # ReLU contains no learnable parameters, so the positive values aren't scaled and are left as is
        self.Relu_0 = nn.ReLU()
        # Sort of kind of does lateral inhibition
        # Same input/output weight dimensions
        # Just with the values adjusted
        self.LRN_0 = nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1)
        # Pooling operates separately on each convolved feature map
        # Does it cut down the number of nodes or leave it intact?
        # when kernel_size == stride, each weight is included in only one pool
        # thus downsampling by a factor of kernel_size (or stride)
        # In the past, pooling was the "trendy" way of downsampling (to reduce training computations)
        # while retaining some kind of translation invariance for the incoming features
        # but pooling is neither necessary nor sufficient for those two
        self.MaxPool_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # Do it all again but with different convolutional dimensions
        # channels per "pixel" going up, n "pixels" actually staying the same
        self.Conv_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.Relu_1 = nn.ReLU()
        self.LRN_1 = nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # and again, a few rounds of channels going up, pixels staying the same
        self.Conv_2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.Relu_2 = nn.ReLU()
        self.Conv_3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2)
        self.Relu_3 = nn.ReLU()
        self.Conv_4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2)
        self.Relu_4 = nn.ReLU()
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.Conv_5 = nn.Conv2d(256, 4096, kernel_size=6, stride=1)
        self.Relu_5 = nn.ReLU()
        # a conv layer becomes fully connected when kernel_size and stride are both 1?
        # I would think it was fully connected once there's only one pixel left...
        # TODO: Confirm that this layer has only one pixel
        self.Conv_6 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1)
        self.Relu_6 = nn.ReLU()
        # Make sure this outputs 20 channels on one pixel
        # AlexNet that comes with torchvision has this as a Linear module, not Conv2d
        # I feel like they are equivalent when the nodes coming in are 1x1px
        # but this feels hackier somehow. Matlab! Dum dum!
        self.Conv_7 = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)
        self.Flatten_0 = nn.Flatten()
        self.Softmax_0 = nn.Softmax(dim=-1)
            
        # TODO: What happens when ReLU is inplace vs not (as ONNX import)?
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        # Don't pass through the initializers module because... it doesn't DO anything to the data
        x = x.to(torch.float)
        x = self.Conv_0(x)
        x = self.Relu_0(x)
        x = self.LRN_0(x)
        x = self.MaxPool_0(x)
        x = self.Conv_1(x)
        x = self.Relu_1(x)
        x = self.LRN_1(x)
        x = self.MaxPool_1(x)
        x = self.Conv_2(x)
        x = self.Relu_2(x)
        x = self.Conv_3(x)
        x = self.Relu_3(x)
        x = self.Conv_4(x)
        x = self.Relu_4(x)
        x = self.MaxPool_2(x)
        x = self.Conv_5(x)
        x = self.Relu_5(x)
        x = self.Conv_6(x)
        x = self.Relu_6(x)
        x = self.Conv_7(x)
        x = self.Flatten_0(x)
        x = self.Softmax_0(x)
        return x

# %%
# Define hook functions to get intermediate activations
# Many thanks to https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/

activations = {}
def cache_layer_activation(name):
    def hook(model, input, output):
        # TODO: Figure out how to get this to return the detached output
        # instead of doing this as a side effect
        activations[name] = output.detach().numpy()
    return hook

# %%
# Load that puppy in
emonet_torch = EmoNet()
emonet_torch.load_state_dict(state_dict=torch.load(emonet_path))

# Decorate a call with @torch.no_grad() to Turn OFF gradient tracking because we don't need to train new weights
# parameters() is an Iterator object,
# which means you can do shit to each element in a for loop
# but you can't necessarily index into each element of it
for param in emonet_torch.parameters():
    param.requires_grad = False

# Register the dang hooks to pull intermediate activations
for name, mod in emonet_torch.named_modules():
    if name.startswith('Conv'):
        mod.register_forward_hook(cache_layer_activation(name))

# %%
# Preload ze NSD image data
# Use the NSD images, not the original COCO images/API, even though they're in torchvision
# because the NSD images come pre-cropped and with their own metadata
# It doesn't take that long because HDF5 lazy-load! Yay, I think
# Gotta resize to 227x227 because that's how AlexNet from Matlab likes things
# But do NOT rescale RGB to 0-1 because AlexNet from Matlab likes 0-255
# TODO: Figure out if we need to use any of the image caching I saw in that guy's GitHub Gist
nsd_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.PILToTensor()
                        ])

nsd_torchdata = NSDDataset(root=nsd_path+'stimuli/',
                           annFile=nsd_path+'nsd_stim_info_merged.csv',
                           shared1000=True,
                           transform=nsd_transform)

# %%
# Initialize ze DataLoader to actually feed into the model

#
batch_size = 10
nsd_torchloader = torch.utils.data.DataLoader(nsd_torchdata, batch_size=batch_size)
# %%
# DO IT?! RUN IT! GET PREDS
emonet_torch.eval()

# Yeah I think we should set this to empty dict again just in case
# It needs to be initalized earlier for that function to work I think
# but for safety, re-set it to empty before every pass of the model
# TODO: Write this to preallocate so we can copy into Tensor slice and go more pythonically
activations_all = {
    'Conv_0': [],
    'Conv_1': [],
    'Conv_2': [],
    'Conv_3': [],
    'Conv_4': [],
    'Conv_5': [],
    'Conv_6': [],
    'Conv_7': [],
    }
activations = {}
preds_all = []
pred = []

for img, lab in tqdm(nsd_torchloader):
    pred = emonet_torch(img)
    preds_all.append(pred.numpy())
    for layer in activations_all.keys():
        activations_all[layer].append(activations[layer])

    # Clear out activations before next img
    activations = {}


# %%
