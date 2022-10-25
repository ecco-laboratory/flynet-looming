#!/bin/bash
#SBATCH --account=default
#SBATCH --partition=day-long
#SBATCH --nodelist=node1
#SBATCH -n 1
#SBATCH --time=0-23:59:59
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --job-name=dtw_flow_matrix
# ------------------
cd ~/Repos/emonet-py
conda activate ./env
python ./python/batch_calc_video_dtw.py