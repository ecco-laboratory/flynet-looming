# %%
# importz
import os
import platform

import av
import cv2 as cv
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import reshape
from PIL import ImageOps
from tqdm import tqdm, trange
from tslearn.metrics import dtw

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

# Need to set repo path because I suspect
# when slurm runs this it doesn't immediately know what is up
repo_path = '/home/mthieu/Repos/emonet-py/'
output_path = os.path.join(repo_path, 'ignore', 'outputs')
local_ck_path = '/home/mthieu/Repos/CowenKeltner'
metadata_path = os.path.join(local_ck_path, 'metadata')

these_classes = ['Anxiety', 'Excitement', 'Fear', 'Horror', 'Surprise']
# %%
# Functions for numpy videos

# io handling
# path = os.path.join(local_ck_path, 'videos_10fps', 'Excitement', '1838.mp4')
def read_and_resize_video_frames(path):
    frames = []
    with av.open(path) as video_container:
        for frame in video_container.decode(video=0):
            # go to PIL Image first to use the resize method
            frame_resized = frame.to_image().resize((227, 227))
            frames.append(np.array(frame_resized))

    frames = np.stack(frames, axis=0)

    return frames

# flatten out every other dimension except time
# so it looks like a "multivariate time series"
def reshape_video_array(array):
    return array.reshape((array.shape[0], -1))

# %%
# Define functions to get optical flow?!

def read_and_calc_video_flow(path):

    flow_frames = []

    # openCV VideoCapture objects SUCK!
    # Specifically, I can't seem to change the auto-threading
    # And I don't want to use every single core, sorry
    # So, try using PyAV's reading functions (which still use ffmpeg under the hood)
    # bc I think the cv2 functions will work on numpy arrays
    # as can be returned by av

    prev_frame_gray = None

    with av.open(path) as video_container:
        for frame in video_container.decode(video=0):
            # go to PIL Image first to use the resize method
            frame_resized = frame.to_image().resize((227, 227))
            frame_gray = ImageOps.grayscale(frame_resized)
            # Just to be sure, I think cv2 needs it as a numpy array
            frame_gray = np.array(frame_gray)

            if prev_frame_gray is None:
                # Fencepost case for the first frame
                prev_frame_gray = frame_gray
            else:
                
                # Calculates dense optical flow by Farneback method
                # Uhh... the parameter values are all the ones from the tutorial
                # Outputs dims of height x width x 2 values (x and y vector, I assume)
                flow = cv.calcOpticalFlowFarneback(
                    prev_frame_gray, frame_gray,
                    None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )

                flow_frames.append(flow)
                prev_frame_gray = frame_gray
    
    flow_frames = np.stack(flow_frames, axis=0)
    return flow_frames

# %%
# Get video metadata for just these classes

# Need these because they have the top-1 human class labels attached to the video IDs
k19_train = pd.read_csv(os.path.join(metadata_path, 'train_video_ids.csv'), index_col='video')
k19_test = pd.read_csv(os.path.join(metadata_path, 'test_video_ids.csv'), index_col='video')

# Truly I wish this was in long form but Alan doesn't like tidy data does he
ck_censored = pd.read_csv(os.path.join(metadata_path, 'censored_video_ids.csv'))
# I'll just get it into a long list
ck_censored = ck_censored['less.bad'].to_list() + ck_censored['very.bad'].to_list()
ck_censored = [vid for vid in ck_censored if not pd.isna(vid)]

video_metadata = pd.read_csv(os.path.join(metadata_path, 'video_10fps_metadata.csv'), index_col='video')
# Have to drop duplicates using the index, not using .drop_duplicates(), because that ignores the index
# and only finds duplicates on the row values
video_metadata = video_metadata[~video_metadata.index.duplicated()]

video_metadata = video_metadata.join(pd.concat([k19_train, k19_test]), how='inner')
video_metadata['train'] = video_metadata.index.isin(k19_train.index)
video_metadata['censored'] = video_metadata.index.isin(ck_censored)

video_metadata = video_metadata[video_metadata['emotion'].isin(these_classes)]
video_metadata = video_metadata[['emotion', 'train', 'censored']]
video_metadata = video_metadata.sort_values(['emotion', 'video'])
# %%
# Calculate straight-up visual similarity between the videos

def calc_video_dtw_matrix(video_fun, save_filename):

    dtw_distances = np.zeros((len(video_metadata), len(video_metadata)))

    # Do the first one reversed to work the longest (bottom) row first

    for idx_row in trange(len(video_metadata), desc='Whole matrix', position=1):
        for idx_col in trange(idx_row, desc='Current row', position=0, leave = False):
            frames_row = video_fun(
                os.path.join(
                    local_ck_path,
                    'videos_10fps',
                    video_metadata['emotion'][idx_row],
                    video_metadata.index[idx_row]
                )
            )
            frames_col = video_fun(
                os.path.join(
                    local_ck_path,
                    'videos_10fps',
                    video_metadata['emotion'][idx_col],
                    video_metadata.index[idx_col]
                )
            )

            frames_row = reshape_video_array(frames_row)
            frames_col = reshape_video_array(frames_col)

            dtw_distances[idx_row, idx_col] = dtw(frames_row, frames_col)
    
    if save_filename is not None:
        np.save(os.path.join(output_path, save_filename), dtw_distances)

    return dtw_distances