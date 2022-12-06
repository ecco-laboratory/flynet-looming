# %%
# importz
import os
import platform

import av
import cv2 as cv
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import reshape
from PIL import Image, ImageOps
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

# resize needs to be a tuple of width, height
def read_and_resize_video_frames(path, resize = None):
    frames = []
    with av.open(path) as video_container:
        for frame in video_container.decode(video=0):
            # go to PIL Image first to use the resize method
            if resize is not None:
                frame_resized = frame.to_image().resize(resize)
            else:
                frame_resized = frame.to_image()
            frames.append(np.array(frame_resized))

    frames = np.stack(frames, axis=0)

    return frames

# flatten out every other dimension except time
# so it looks like a "multivariate time series"
def reshape_video_array(array):
    return array.reshape((array.shape[0], -1))

# %%
# Define functions to work with optical flow?!

def get_pixel_radius_vector(x, y, width, height):
    # WORKS BEST WHEN THERE IS AN ACTUAL PIXEL AT THE CENTER!
    # returns the unit-circle xy vector for a pixel's location on its unit circle
    # centered on the center pixel

    # - 1 because python does its 0 indexing bullshit
    x_center = np.ceil(width / 2 - 1)
    y_center = np.ceil(height / 2 - 1)

    run = x - x_center
    # This one is flipped so that 0 is at the top
    rise = y_center - y

    if run == 0 and rise == 0:
        # Special case at the center
        # to avoid dividing by hypotenuse of 0
        # Will feed forward as always showing 0 radial flow at the center
        return np.array([[0, 0]])
    else:
        # Can you believe I am using the Pythagorean theorem?
        hypotenuse = np.sqrt(run**2 + rise**2)
        # Dividing the hypotenuse (radius) by itself would unit-ify the radius
        # So scaling down both of the other legs by the hypotenuse
        # should return the correct x-y components of the unit circle vector
        # Returns a 1x2 array
        return np.array([[run/hypotenuse, rise/hypotenuse]])

def get_pixel_radial_flow(flow_vector, rad_vector):
    # Now takes the dot product!
    # thanks Phil for reminding me that linear algebra exists
    # As long as the radius vector has already been normalized to unit magnitude
    # This should work

    # Assuming both come in as 1x2 arrays,
    # Transpose the second one to 2x1
    # so you can use @ for matrix multiplication
    # as recommended by numpy

    # Returns JUST the magnitude
    # bc dot product returns a scalar
    return float(flow_vector @ rad_vector.T)

def calc_radial_flow(frames):
    # Returns ONLY frames of magnitudes
    # Because the whole point of radial flow is that the angle is presumed to be... radial
    n_frames = frames.shape[0]
    frame_width = frames.shape[1]
    frame_height = frames.shape[2]
    # Can't use zeros_like because we only need length 1 on the 4th dimension
    # bc pixelwise radial flow comes out as a magnitude
    radial_flow_frames = np.zeros((n_frames, frame_width, frame_height))
    # Can't FULLY vectorize this because we need to operate on the pairs in the 4th dimension together
    
    for x in range(frames.shape[1]):
        for y in range(frames.shape[2]):
            # Don't need to use cartToPolar
            # bc dot products work on cartesian vectors!
            # Calculate this BEFORE iterating over the time dimension
            # because this will be the same for a given pixel location at every timepoint
            pixel_rad_vector = get_pixel_radius_vector(
                x=x,
                y=y,
                width=frame_width,
                height=frame_height
            )
            for time in range(frames.shape[0]):
                radial_flow_frames[time, x, y] = get_pixel_radial_flow(
                    flow_vector=frames[time, x, y, :],
                    rad_vector=pixel_rad_vector
                )
    
    return radial_flow_frames
    

def read_and_calc_video_flow(path, resize = None):

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
            frame_resized = frame.to_image()
            if resize is not None:
                # go to PIL Image first to use the resize method
                frame_resized = frame_resized.resize(resize)
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
                    0.5, 3, 13, 3, 5, 1.1, 0
                )

                flow_frames.append(flow)
                prev_frame_gray = frame_gray
    
    flow_frames = np.stack(flow_frames, axis=0)
    return flow_frames

def convert_flow_to_rgb_polar(frames):
    frames_converted = []
    for frame in frames:
        frame_converted = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        magnitude, angle = cv.cartToPolar(frame[..., 0], frame[..., 1])
        # HUE from the angle, with red at 0 degrees
        frame_converted[..., 0] = angle * 180 / np.pi / 2
        # SATURATION set to max for party vibes
        frame_converted[..., 1] = 255
        # VALUE from the magnitude so lighter = more flow
        frame_converted[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        frame_converted = Image.fromarray(frame_converted, mode='HSV')
        frame_converted = np.array(frame_converted.convert(mode='RGB'))
        frames_converted.append(frame_converted)

    frames_converted = np.stack(frames_converted, axis=0)
    return frames_converted

def convert_flow_to_rgb_cart(frames):
    frames_converted = []
    for frame in frames:
        frame_converted = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        # Dumb Bitch Normalization:
        # multiply/divide by a constant and then add/subtract another constant
        # so that original "flow unit" information is still maintained
        # I am going to hope to jesus that no abs(flow) is ever bigger than 512
        # Because that is basically hard coded in here
        # code x to red
        frame_converted[..., 0] = (frame[..., 0] / 4) + 128
        # nothing to green cuz green sux
        frame_converted[..., 1] = 0
        # code y to blue
        frame_converted[..., 2] = (frame[..., 1] / 4) + 128

        frames_converted.append(frame_converted)

    frames_converted = np.stack(frames_converted, axis=0)
    return frames_converted

def write_arrays_to_imgs(frames, filename_stem):
    # Assumes frames are stored as ndarrays
    # So converts to pillow Image first
    for frame_id in range(len(frames)):
        img = Image.fromarray(frames[frame_id])
        # There should never be more than 110 frames in our CK 10fps set
        # so padding to 3 digits should be sufficient for that
        # But for other videos... who knows. So, to be safe
        # For UCF101, adding that f prefix before the flow-frame number
        # to be consistent with its apparent existing naming convention
        img.save(os.path.join(filename_stem+f'_f{frame_id:04}'+'.jpeg'))
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