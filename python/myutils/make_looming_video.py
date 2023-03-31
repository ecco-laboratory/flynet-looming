# %%
# imports
import os
import platform
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import trange

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
stimuli_path = os.path.join(repo_path, 'ignore', 'stimuli', 'looming_homemade')

# %%
# Argle parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-f',
    '--file',
    type=str,
    help="Source image name (in the looming folder) with file ending"
)
parser.add_argument(
    '-d',
    '--duration',
    type=int,
    help="Looming video duration in seconds"
)
args = vars(parser.parse_args())

# %%
# Testing out img reading and resizing stuff
img = Image.open(os.path.join(stimuli_path, 'images', args['file']))
# turn every transparent pixel to fully opaque
# leaves the actual object intact but turns the background to black
img.putalpha(255)
# and then toss the alpha channel
img = img.convert('RGB')
# Assume constant velocity approaching object
# TODO:
# 1. Compose framewise function that outputs object diameter as a function of velocity
# 2. Loop that function over frames
# 3. Write those frames as a video

# per Sun and Frost 1998 appendix (THANK GOD),
# theta_t = 2 * arctan(diameter_initial / (2 * distance_t))
# use this to calculate theta_t on every frame
# then re-project the on-screen diameter 
# using theta_t and the STARTING (viewing) distance

# if 25 fps, each frame is 40 ms
diameter_initial = 100 # in px, basically arbitrary bc this would be "real world" obj size
distance_initial = diameter_initial * 18 # 18 obj-widths away from screen? sure
duration_sec = args['duration'] # in seconds
fps = 25
duration_frames = duration_sec * fps # use int division so range gets an int
# velocity in px/frame, weird units but I think it will work for now
# assuming collision occurs at the end of the video duration
velocity = distance_initial / duration_frames
scale = diameter_initial / img.width
frames = []
# literally the monitor size in my office, just for testing
screen_size = (1920, 1080)

for frame in trange(duration_frames):
    # current distance = initial distance - distance traveled since then
    distance = distance_initial - velocity*frame
    # first calculate theta using the fake distance of the looming object
    theta = 2 * np.arctan2(diameter_initial / 2, distance)
    # then calculate the drawn diameter reapplying theta back to the screen distance
    diameter = 2 * np.tan(theta / 2) * distance_initial
    scale = diameter / img.width
    # use ImageOps.scale instead of Image.resize bc scale always retains aspect ratio
    this_img = ImageOps.scale(img, scale)
    # if width is larger than screen width, crop width but leave height the same
    if this_img.width > screen_size[0]:
        this_img = this_img.crop((
            (this_img.width-screen_size[0])/2, 
            0, 
            (this_img.width+screen_size[0])/2, 
            this_img.height
        ))
    # and vice versa for height
    if this_img.height > screen_size[1]:
        this_img = this_img.crop((
            0, 
            (this_img.height-screen_size[1])/2, 
            this_img.width, 
            (this_img.height+screen_size[1])/2
        ))
    # if smaller than screen, pad it out to screen size
    # must do this by first expand()-ing the "wider" dimension equal to screen size
    # "wider" being relative to screen aspect ratio
    if img.width/img.height >= screen_size[0]/screen_size[1]:
        this_img = ImageOps.expand(this_img, (screen_size[0]-this_img.width)//2, 0)
    else:
        this_img = ImageOps.expand(this_img, (screen_size[1]-this_img.height)//2, 0)
    this_img = ImageOps.pad(this_img, screen_size)
    frames.append(this_img)
# %%
# export frames as video he he he
# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/

container = cv2.VideoWriter(
    os.path.join(stimuli_path, 'videos', os.path.splitext(args['file'])[0] + '.mp4'),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    screen_size
)

for frame in trange(len(frames)):
    # flip from RGB to BGR. So STUPID
    container.write(np.array(frames[frame])[:, :, ::-1])
container.release()
