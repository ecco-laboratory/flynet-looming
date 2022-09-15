# %%
# imports, per usual

import onnx
import torch
from onnx import helper, version_converter
from onnx2torch import convert

emonet_path = '../ignore/models/'
# %%
# load the ONNX version of the model
emonet_onnx = onnx.load(emonet_path+'EmoNet.onnx')
emonet_onnx_13 = version_converter.convert_version(emonet_onnx, 13)
emonet_onnx2torch = convert(emonet_onnx_13)
# %%
# write the model dict out
torch.save(emonet_onnx2torch.state_dict(), emonet_path+'EmoNet.pt')
