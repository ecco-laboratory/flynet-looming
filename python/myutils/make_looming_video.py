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
# stimuli_path = os.path.join(repo_path, 'ignore', 'stimuli', 'looming_homemade')

# %%
# Argle parser
parser = ArgumentParser(
    prog='make_looming_video',
    description='Create a looming video from an image.',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--infile',
    type=str,
    help="Path to source image"
)
parser.add_argument(
    '--outpath',
    type=str,
    help="Path to destination video FOLDER, EXCLUDING filename (will be auto-generated)"
)
parser.add_argument(
    '--diststart',
    default=18,
    type=float,
    help="Distance that object should START AT, in object-widths"
)
parser.add_argument(
    '--distend',
    default=0,
    type=float,
    help="Distance that object should END AT, in object-widths"
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
parser.add_argument(
    '--fps',
    default=25,
    type=int,
    help="Frames per second of output video"
)
args = vars(parser.parse_args())

# %%
# Helper functions for the loom resizer
def scale_image_by_viewing_distance(image, diameter_initial, distance_initial, distance_current):
    # per Sun and Frost 1998 appendix (THANK GOD),
    # theta_t = 2 * arctan(diameter_initial / (2 * distance_t))

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
img = Image.open(args['infile'])
# turn every transparent pixel to fully opaque
# leaves the actual object intact but turns the background to black
img.putalpha(255)
# and then toss the alpha channel
img = img.convert('RGB')
# Assume constant velocity approaching object

# if 25 fps, each frame is 40 ms
diameter_initial = args['objwidth'] # in px, basically arbitrary bc this would be "real world" obj size
distance_initial = diameter_initial * args['diststart'] # Default 18 obj-widths away from screen? sure
distance_final = diameter_initial * args['distend'] # default 0 obj-widths, or "full collision"
direction = args['direction']
duration_wait_sec = args['pausetime'] # in seconds
duration_loom_sec = args['loomtime'] # in seconds
fps = args['fps']
duration_wait_frames = np.floor(duration_wait_sec * fps).astype(int)
# use ceil for looming duration to round up and get the image "closer" to viewer
duration_loom_frames = np.ceil(duration_loom_sec * fps).astype(int)
# velocity in px/frame, weird units but I think it will work for now
# assuming very-near collision occurs at the end of the video duration
# the arbitrary looking adjustment is to get it so distance is never 0
# because that causes PIL to overflow when attempting to resize the image to some HUGE size
velocity = ((distance_initial-distance_final)-10) / (duration_loom_frames-1)
# default is literally the monitor size in my office, just for testing
screen_size = tuple(args['framesize'])

# Initialize the frames as empty first
frames = []

# Generate the actual motion sequence
for frame in trange(duration_loom_frames, desc='Expanding images to loom'):
    # current distance = initial distance - distance traveled since then
    distance = distance_initial - velocity*frame
    this_img = scale_image_by_viewing_distance(img, diameter_initial, distance_initial, distance)
    this_img = fit_image_to_frame(this_img, screen_size)
    frames.append(this_img)

# Put my thang down flip it and reverse it
if direction == 'receding':
    frames.reverse()

# Put the waiting frames on
# This way it always waits at the beginning of the video itself
# So for receding videos it waits at the close-up
if duration_wait_frames > 0:
    wait_frames = [frames[0]] * duration_wait_frames
    frames = wait_frames + frames
# %%
# export frames as video he he he
# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
video_name = [
    os.path.splitext(os.path.split(args['infile'])[1])[0], 
    direction, 
    'pause' + '{:02.1f}'.format(args['pausetime']), 
    'loom' + '{:02.1f}'.format(args['loomtime'])
]
video_name = '_'.join(video_name) + '.mp4'
container = cv2.VideoWriter(
    os.path.join(args['outpath'], video_name),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    screen_size
)

for frame in trange(len(frames), desc='Writing frames to video'):
    # flip from RGB to BGR. So STUPID
    container.write(np.array(frames[frame])[:, :, ::-1])
container.release()
