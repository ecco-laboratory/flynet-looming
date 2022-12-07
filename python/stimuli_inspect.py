# %%
# imports, whateva
import os

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from video_dtw_utils import (calc_radial_flow, local_ck_path, output_path,
                             read_and_calc_video_flow, video_metadata)


def matrix_tri_to_sym(matrix):
    # Turn it from bottom-half to symmetric matrix
    # I think the scipy clustering called by seaborn demands it
    # This works because the empty half is 0s, not NaNs
    return matrix + matrix.T
# %%
# Attempt to visualize raw image similarity with frickin seaborn
# Get that big ass numpy array into a square pandas dataframe with both nested col header and row indices

dtw_raw = np.load(os.path.join(output_path, 'dtw_distances_5class_raw.npy'))

dtw_raw = matrix_tri_to_sym(dtw_raw)

sns.clustermap(
    pd.DataFrame(
        dtw_raw,
        index=video_metadata['emotion'],
        columns=video_metadata['emotion']
    )
)

# %%
# Visualize flow image similarity? yeah??
dtw_flow = np.load(os.path.join(output_path, 'dtw_distances_5class_flow.npy'))
dtw_flow = matrix_tri_to_sym(dtw_flow)

sns.clustermap(
    pd.DataFrame(
        dtw_flow,
        index=video_metadata['emotion'],
        columns=video_metadata['emotion']
    )
)
# %%
# Calculate avg radial flow for every VIDEO
radial_flows = []

for idx, row in tqdm(video_metadata.iterrows()):
    path = os.path.join(local_ck_path, 'videos_10fps', row['emotion'], idx)
    flow_frames = read_and_calc_video_flow(path)
    radial_flow_frames = calc_radial_flow(flow_frames)
    # Mean across the whole video
    mean_radial_flow = np.mean(radial_flow_frames)
    radial_flows.append(mean_radial_flow)

radial_flow_df = video_metadata
radial_flow_df['radial_flow'] = radial_flows

radial_flow_df.to_csv(os.path.join(output_path, 'ck2017_5class_radial_flows.csv'))
# %%
