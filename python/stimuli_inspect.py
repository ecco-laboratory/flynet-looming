# %%
# imports, whateva
import os

import numpy as np
import pandas as pd
import seaborn as sns
from video_dtw_utils import output_path, video_metadata


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
