# %%
# imports

import os
import platform
from fractions import Fraction

import ffmpeg
import pandas as pd
from tqdm import tqdm

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

stim_path = os.path.join(base_path, 'CowenKeltner/Videos_by_Category')
metadata_path = '/home/mthieu/Repos/CowenKeltner/metadata'

# %%
# extract metadata for videos
# This takes like 1s per video! Not trivial with 2k videos
video_paths = []
video_ids = []
video_heights = []
video_widths = []
video_frame_rates = []
video_durations = []
video_nframes = []
for root, pwd, files in os.walk(stim_path):
    print('Current folder:', root)
    for file in tqdm(files):
        if file.endswith('.mp4'):
            video_ids.append(file)
            path = os.path.join(root, file)
            video_paths.append(path)

            if False:
                metadata = ffmpeg.probe(path)['streams'][0]
                video_heights.append(metadata['height'])
                video_widths.append(metadata['width'])
                video_frame_rates.append(metadata['r_frame_rate'])
                video_durations.append(metadata['duration'])
                video_nframes.append(metadata['nb_frames'])

# %%
# Bind said metadata into a dataframe
video_metadata = pd.DataFrame({
    'video': video_ids,
    'height': video_heights,
    'width': video_widths,
    'frame_rate': video_frame_rates,
    'duration': video_durations,
    'nframes': video_nframes})
video_metadata = video_metadata.set_index('video')
video_metadata['frame_rate_float'] = [float(Fraction(rate)) for rate in video_metadata['frame_rate']]

video_metadata.to_csv(os.path.join(metadata_path, 'video_metadata.csv'))

# %%
# Inspecting the metadata

video_metadata = pd.read_csv(os.path.join(metadata_path, 'video_metadata.csv'), index_col='video')

# Need these because they have the top-1 human class labels attached to the video IDs
k19_train = pd.read_csv(os.path.join(metadata_path, 'train_video_ids.csv'), index_col='video')
k19_test = pd.read_csv(os.path.join(metadata_path, 'test_video_ids.csv'), index_col='video')

# Truly I wish this was in long form but Alan doesn't like tidy data does he
ck_censored = pd.read_csv(os.path.join(metadata_path, 'censored_video_ids.csv'))
# I'll just get it into a long list
ck_censored = ck_censored['less.bad'].to_list() + ck_censored['very.bad'].to_list()
ck_censored = [vid for vid in ck_censored if not pd.isna(vid)]

video_metadata = video_metadata.join(pd.concat([k19_train, k19_test]))
video_metadata['censored'] = video_metadata.index.isin(ck_censored)
# %%
# Copy videos out at 10 fps

fps10_path = '/home/mthieu/Repos/CowenKeltner/videos_10fps'
for idx, row in tqdm(video_metadata.query('duration <= 11').iterrows()):
    emo_orig_path = os.path.join(stim_path, row['emotion'])
    emo_fps10_path = os.path.join(fps10_path, row['emotion'])

    if not os.path.exists(emo_fps10_path):
        os.makedirs(emo_fps10_path)

    if not os.path.isfile(os.path.join(emo_fps10_path, idx)):
        (
            ffmpeg
            .input(os.path.join(emo_orig_path, idx))
            .output(os.path.join(emo_fps10_path, idx), r=10)
            .overwrite_output()
            .run()
        )

# %%
