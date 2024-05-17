# %%
# imports up top, babies

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
import torch
import torchvision
from myutils.emonet_utils import Cowen2017Dataset, EmoNet, emonet_output_classes
from tqdm import tqdm
# %%
# Argle parser
parser = ArgumentParser(
    prog='calc_emonet_activations',
    description='Get PyTorch EmoNet predictions for a directory of videos.',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '-i',
    '--in_folder',
    type=str,
    help="Path to source video FOLDER. This folder is the source for the Dataset object."
)
parser.add_argument(
    '-o',
    '--out_path',
    type=str,
    help="Path to write out overall activations file for all videos input"
)
parser.add_argument(
    '-w',
    '--weight_path',
    type=str,
    help="Which model path to pull pre-trained EmoNet weights from?"
)
parser.add_argument(
    '-m',
    '--metadata_path',
    type=str,
    help="Which CSV path to pull Cowen 2017/Kragel 2019 emotion labels from?"
)

args = vars(parser.parse_args())

video_path = args['in_folder']
out_path = args['out_path']
model_path = args['weight_path']
metadata_path = args['metadata_path']

# %%
# Fire up the full EmoNet
emonet_torch = EmoNet()
emonet_torch.load_state_dict(state_dict=torch.load(model_path))

# Turn off backward gradients for all runs of the model
# Right now we just be inferencing
for param in emonet_torch.parameters():
    param.requires_grad = False

# %%
# read those motherfuckin videos in
ck_torchdata_test = Cowen2017Dataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    train=False,
    transform=torchvision.transforms.Resize((227, 227))
)

# Set batch_size here to 1 so it's just one video at a time
# BUT! Each video effectively acts as a batch of frames, as long as time is in the first dim
ck_torchloader_test = torch.utils.data.DataLoader(ck_torchdata_test, batch_size=1)
# %%
# Let's get predicting (full EmoNet)
emonet_torch.eval()

preds_all = {}

# For full EmoNet, only need to do it on the test data 
# because it's been trained on the training data (duh)
for vid, lab in tqdm(ck_torchloader_test):
    vid = vid.squeeze()
    pred = emonet_torch(vid)
    preds_all[lab['id'][0]] = pred.numpy()

# convert the frames x emotions predictions into dataframes
# and concatenate for writing out
for id in preds_all.keys():
    preds_all[id] = pd.DataFrame(preds_all[id],
                                 columns=emonet_output_classes)

preds_all = pd.concat(preds_all, names=['video', 'frame']).reset_index(level='frame')

preds_all.to_csv(out_path)
# %%
