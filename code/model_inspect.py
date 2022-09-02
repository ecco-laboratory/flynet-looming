# %%
# imports, per usual
import torch
import onnx
from onnx import version_converter, helper
import onnx2torch

emonet_path = '../ignore/models/EmoNet.onnx'
# %%
# load the ONNX version of the model
emonet_onnx = onnx.load(emonet_path)
emonet_onnx_13 = version_converter.convert_version(emonet_onnx, 13)
# emonet_torch = convert(emonet_onnx_13)
# %%
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# %%
