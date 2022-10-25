# %%
# importz

from video_dtw_utils import (calc_video_dtw_matrix, read_and_calc_video_flow,
                             read_and_resize_video_frames, reshape_video_array)

# %%
# Dewit
# Not assigning output to variable because the point is the file write anyway
calc_video_dtw_matrix(video_fun=read_and_calc_video_flow, save_filename='dtw_distances_5class_flow.npy')

# %%
