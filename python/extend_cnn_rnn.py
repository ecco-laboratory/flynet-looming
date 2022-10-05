# %%
# imports up top, babies

import os
import platform

import torch
import torchvision
from emonet_utils import (Cowen2017Dataset, EmoNet, EmoNetHeadlessVideo,
                          GRUClassifier)
from image_utils import pad_sequence_tuple
from torch.nn.utils import rnn as rnn_utils
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
ck_torchdata = Cowen2017Dataset(root=os.path.join(local_ck_path, 'videos_10fps'),
                                annPath=os.path.join(local_ck_path, 'metadata'),
                                train=False,
                                transform=torchvision.transforms.Resize((227, 227))
                                )

# Mind you: because this routes the videos through pad_sequence() in order to batch them,
# it also returns a list of the nframes of each video in the batch
ck_torchloader = torch.utils.data.DataLoader(ck_torchdata, batch_size=7, collate_fn=pad_sequence_tuple)
# %%
# Get stage 1 preds (just for now)
emonet_headless.eval()

for vids, lens, labs in tqdm(ck_torchloader):
    vids_packed = rnn_utils.pack_padded_sequence(
        input=vids,
        lengths=lens,
        batch_first=True,
        enforce_sorted=False
    )
    preds = emonet_headless(vids_packed.data)
    preds_packed = rnn_utils.PackedSequence(
        data=preds,
        batch_sizes=vids_packed.batch_sizes,
        sorted_indices=vids_packed.sorted_indices,
        unsorted_indices=vids_packed.unsorted_indices
    )

# %%
# Testing out the GRU output
test_big_gru = GRUClassifier()
