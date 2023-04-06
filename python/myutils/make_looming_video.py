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
parser = ArgumentParser(
    prog='make_looming_video',
    description='Create a looming video from an image.',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--file',
    type=str,
    help="Source image name (in the looming folder) with file ending"
)
parser.add_argument(
    '--direction',
    default='looming',
    type=str,
    choices=['looming', 'receding'],
    help="Type of motion? Looming (forwards) or receding (backwards)? Defaults to looming"
)
parser.add_argument(
    '--pausetime',
    default=0,
    type=float,
    help="Pre-looming pause (static) duration in seconds"
)
parser.add_argument(
    '--loomtime',
    default=1,
    type=float,
    help="Looming (active motion) duration in seconds"
)
parser.add_argument(
    '--objwidth',
    default=100,
    type=int,
    help="Desired starting image width in pixels"
)
parser.add_argument(
    '--framesize',
    default=[1920, 1080],
    type=int,
    nargs=2,
    help="Output video frame size, as 2 pixel values (width, then height)"
)
args = vars(parser.parse_args())

# %%
# Helper functions for the loom resizer
def scale_image_by_viewing_distance(image, diameter_initial, distance_initial, distance_current):
    # first calculate theta using the fake distance of the looming object
    theta = 2 * np.arctan2(diameter_initial / 2, distance_current)
    # then calculate the drawn diameter reapplying theta back to the screen distance
    diameter = 2 * np.tan(theta / 2) * distance_initial
    scale = diameter / img.width
    # use ImageOps.scale instead of Image.resize bc scale always retains aspect ratio
    image = ImageOps.scale(image, scale)
    return image

def fit_image_to_frame(image, frame_size):
    # first calculate aspect ratio of the input image before it gets resized
    aspect_ratio = image.width/image.height
    # if width is larger than screen width, crop width but leave height the same
    if image.width > frame_size[0]:
        image = image.crop((
            (image.width-frame_size[0])/2, 
            0, 
            (image.width+frame_size[0])/2, 
            image.height
        ))
    # and vice versa for height
    if image.height > frame_size[1]:
        image = image.crop((
            0, 
            (image.height-frame_size[1])/2, 
            image.width, 
            (image.height+frame_size[1])/2
        ))
    # if smaller than screen, pad it out to screen size
    # must do this by first expand()-ing the "wider" dimension equal to screen size
    # "wider" being relative to screen aspect ratio
    if aspect_ratio >= frame_size[0]/frame_size[1]:
        image = ImageOps.expand(image, (frame_size[0]-image.width)//2, 0)
    else:
        image = ImageOps.expand(image, (frame_size[1]-image.height)//2, 0)
    image = ImageOps.pad(image, frame_size)

    return image
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
diameter_initial = args['objwidth'] # in px, basically arbitrary bc this would be "real world" obj size
distance_initial = diameter_initial * 18 # 18 obj-widths away from screen? sure
duration_start_sec = args['pausetime'] # in seconds
duration_loom_sec = args['loomtime'] # in seconds
fps = 25
duration_start_frames = np.floor(duration_start_sec * fps).astype(int)
# use ceil for looming duration to round up and get the image "closer" to viewer
duration_loom_frames = np.ceil(duration_loom_sec * fps).astype(int)
# velocity in px/frame, weird units but I think it will work for now
# assuming very-near collision occurs at the end of the video duration
# the arbitrary looking adjustment is to get it so distance is never 0
# because that causes PIL to overflow when attempting to resize the image to some HUGE size
velocity = (distance_initial-10) / (duration_loom_frames-1)
# default is literally the monitor size in my office, just for testing
screen_size = tuple(args['framesize'])

# START the frames with the 'first' static image
if duration_start_frames > 0:
    frames = [fit_image_to_frame(ImageOps.scale(img, diameter_initial / img.width), screen_size)] * duration_start_frames
else:
    frames = []

for frame in trange(duration_loom_frames, desc='Expanding images to loom'):
    # current distance = initial distance - distance traveled since then
    distance = distance_initial - velocity*frame
    this_img = scale_image_by_viewing_distance(img, diameter_initial, distance_initial, distance)
    this_img = fit_image_to_frame(this_img, screen_size)
    frames.append(this_img)
# %%
# export frames as video he he he
# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
video_name = [
    os.path.splitext(
    args['file'])[0], 
    args['direction'], 
    'pause' + '{:02.1f}'.format(args['pausetime']), 
    'loom' + '{:02.1f}'.format(args['loomtime'])
]
video_name = '_'.join(video_name) + '.mp4'
container = cv2.VideoWriter(
    os.path.join(stimuli_path, 'videos', video_name),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    screen_size
)

if args['direction'] == 'receding':
    frames.reverse()

for frame in trange(len(frames), desc='Writing frames to video'):
    # flip from RGB to BGR. So STUPID
    container.write(np.array(frames[frame])[:, :, ::-1])
container.release()