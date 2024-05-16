# %%
# importz
import os

import av
import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import reshape
from PIL import Image, ImageOps
from tqdm import tqdm

# %%
# Functions for numpy videos

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

def read_and_calc_video_flow(path, resize = None, progress = False):

    flow_frames = []

    # openCV VideoCapture objects SUCK!
    # Specifically, I can't seem to change the auto-threading
    # And I don't want to use every single core, sorry
    # So, try using PyAV's reading functions (which still use ffmpeg under the hood)
    # bc I think the cv2 functions will work on numpy arrays
    # as can be returned by av

    prev_frame_gray = None

    with av.open(path) as video_container:
        if progress:
            iterator = tqdm(video_container.decode(video=0))
        else:
            iterator = video_container.decode(video=0)

        for frame in iterator:
            frame_resized = frame.to_image()
            if resize is not None:
                # go to PIL Image first to use the resize method
                # Image.resize requires a 2-tuple for size!
                # does not assume square when one value is passed in
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

def write_arrays_to_gif(frames, filename, fps=25):
    # Assumes frames are stored as ndarrays
    # So converts to list of pillow Images first
    frames = [Image.fromarray(frames[i]) for i in range(len(frames))]

    frames[0].save(
        filename,
        save_all=True, 
        append_images=frames[1:], 
        optimize=False, 
        duration=1000/fps, 
        loop=0 # this means loop!
    )
    
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
