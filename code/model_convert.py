# %%
# imports, per usual
# including the local version of onnx2torch
# presuming the conda-forge version isn't already installed
# append puts it onto the end of the path (aka lowest priority)
import sys

import onnx
import torch
import torchvision
from onnx import helper, version_converter

sys.path.append('/Users/mthieu/Repos/onnx2torch')
from onnx2torch import convert

emonet_path = '../ignore/models/EmoNet.onnx'
# %%
# load the ONNX version of the model
emonet_onnx = onnx.load(emonet_path)
emonet_onnx_13 = version_converter.convert_version(emonet_onnx, 13)
emonet_torch = convert(emonet_onnx_13)
# %%
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# %%
