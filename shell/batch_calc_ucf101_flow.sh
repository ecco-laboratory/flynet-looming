#!/bin/bash
#SBATCH --account=default
#SBATCH --partition=day-long
#SBATCH --nodelist=node1
#SBATCH -n 1
#SBATCH --time=0-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=write_ucf101_imgs
# ------------------

source ~/.bashrc
cd ~/Repos/emonet-py
source activate ./env

python ./python/extend_fly.py
