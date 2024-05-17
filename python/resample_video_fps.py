# %%
# imports

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import ffmpeg
from tqdm import tqdm

# %%
# Argle parser
parser = ArgumentParser(
    prog='resample_video_fps',
    description='Resample a directory of videos to 10 fps.',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--in_folder',
    '-i',
    type=str,
    help="Path to source video FOLDER. Will attempt to operate over every video in folder."
)
parser.add_argument(
    '--out_folder',
    '-o',
    type=str,
    help="Path to destination video FOLDER. Files will have same name"
)

args = vars(parser.parse_args())

stim_path = args['in_folder']
fps10_path = args['out_folder']

# %%
# Copy videos out at 10 fps
# But only those that are less than or equal to 11 seconds--see below

for root, pwd, files in os.walk(stim_path):
    print('Current folder:', root)
    for file in tqdm(files):
        if file.endswith('.mp4'):
            path = os.path.join(root, file)

            # Need to get video duration from metadata first
            metadata = ffmpeg.probe(path)['streams'][0]
            video_duration = float(metadata['duration'])
            if video_duration <= 11:
                (
                    ffmpeg
                    .input(path)
                    .output(os.path.join(fps10_path, file), r=10)
                    .overwrite_output()
                    .run()
                )

# %%
