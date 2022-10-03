# %%
# imports up top, babies

import os
import platform

import pandas as pd
import torch
import torchvision
from emonet_utils import EmoNet
from image_utils import Cowen2017Dataset
from tqdm import tqdm

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

emonet_path = '../ignore/models/EmoNet.pt'
metadata_path = '/home/mthieu/Repos/CowenKeltner/metadata'

# %%
# Fire up the ol net
emonet_torch = EmoNet()
emonet_torch.load_state_dict(state_dict=torch.load(emonet_path))

# Turn off backward gradients for all runs of the model
# Right now we just be inferencing
for param in emonet_torch.parameters():
    param.requires_grad = False

# %%
# Defining EmoNet's output classes--should do this in a utils though
emonet_output_classes = [
    'Adoration',
    'Aesthetic Appreciation',
    'Amusement',
    'Anxiety',
    'Awe',
    'Boredom',
    'Confusion',
    'Craving',
    'Disgust',
    'Empathic Pain',
    'Entrancement',
    'Excitement',
    'Fear',
    'Horror',
    'Interest',
    'Joy',
    'Romance',
    'Sadness',
    'Sexual Desire',
    'Surprise'
]

# %%
# Read in the Cowen & Keltner metadata

ck_metadata = pd.read_csv(os.path.join(metadata_path, 'video_ratings.csv'))
ck_metadata = ck_metadata.set_index('Filename')

# Truly I wish this was in long form but Alan doesn't like tidy data does he
ck_censored = pd.read_csv(os.path.join(metadata_path, 'censored_video_ids.csv'))

k19_train = pd.read_csv(os.path.join(metadata_path, 'train_video_ids.csv'), index_col='video')
k19_test = pd.read_csv(os.path.join(metadata_path, 'test_video_ids.csv'), index_col='video')

ck_viddata = pd.read_csv(os.path.join(metadata_path, 'video_metadata.csv'), index_col='video')

# We don't need to see the censored ones! At least I personally don't
# I guess the model doesn't have feelings
if True:
    ck_metadata = ck_metadata[~ck_metadata.index.isin(ck_censored['less.bad'])]
    ck_metadata = ck_metadata[~ck_metadata.index.isin(ck_censored['very.bad'])]

# Just the emotion categories
ck_metadata = ck_metadata.iloc[:, range(34)]

ck_metadata = ck_metadata.join(pd.concat([k19_train, k19_test]))
ck_viddata = ck_viddata.join(pd.concat([k19_train, k19_test]))
# %%
# read those motherfuckin videos in

test = torchvision.io.read_video(os.path.join(base_path, 'CowenKeltner/Videos_by_Category/Adoration/0035.mp4'), pts_unit='sec')

ck_torchdata = Cowen2017Dataset(root=os.path.join(base_path, 'CowenKeltner/Videos_by_Category'),
                                annPath=metadata_path,
                                train=False,
                                transform=torchvision.transforms.Resize((227, 227))
                                )

# Set batch_size here to 1 so it's just one video at a time
# BUT! Each video effectively acts as a batch of frames, as long as time is in the first dim
ck_torchloader = torch.utils.data.DataLoader(ck_torchdata, batch_size=1)
# %%
# Let's get predicting
emonet_torch.eval()

preds_all = {}

for vid, lab in tqdm(ck_torchloader):
    vid = vid.squeeze()
    pred = emonet_torch(vid)
    preds_all[lab['id'][0]] = pred.numpy()

# %%
# the framewise predictions are! big dataframe!

for id in preds_all.keys():
    preds_all[id] = pd.DataFrame(preds_all[id], columns=emonet_output_classes)

preds_all = pd.concat(preds_all, names=['video', 'frame'])

preds_all['guess_1'] = preds_all.idxmax(axis=1)

preds_all.to_csv(os.path.join(metadata_path, 'test_framewise_preds.csv'))
# %%
