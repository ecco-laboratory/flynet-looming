#!/usr/bin/env Rscript

# This is a helper script to run the pipeline.
# Choose how to execute the pipeline below.
# See https://books.ropensci.org/targets/hpc.html
# to learn about your options.

# tar_make() as provided below will build each pipeline _in series,_
# one intermediate object at a time. Nothing wrong with this,
# But it will take a while, especially for Study 1 (retinotopy).

# IF YOU DARE! You can attempt to run the pipeline for any study using
# targets::tar_make_future(), with the workers argument specifying
# the number of targets to attempt to make at a time (1 target/worker per slurm job).
# IF YOU DO THIS, you will need to go into their respective _targets_[STUDY].R files
# and edit the slurm parameters set in the resources argument of plan(batchtools_slurm)
# at the top of the file to match the slurm parameters for your cluster BEFORE you run this script.
# In particular:
# 1. Set the nodelist and exclude arguments as you would for any slurm job on your cluster
# and omit them if you can run any job on any node equally.
# I had to exclude some nodes on our cluster that don't have R installed, for example.
# 2. Set the walltime and partition arguments as you would for any slurm job on your cluster
# based on the named partitions and their max runtime limits. Walltime must be set in seconds.
Sys.setenv(TAR_PROJECT='subjective') # Study 3
targets::tar_make()
Sys.setenv(TAR_PROJECT='eyeblink') # Study 2
targets::tar_make()
Sys.setenv(TAR_PROJECT='retinotopy') # Study 1
# Make everything except for the two most memory-hungry targets
targets::tar_make(-c(perms_flynet_v1_studyforrest, statmaps_connectivity_studyforrest))
# Each of these permutation batches runs on a separate worker requiring about 4 GB of memory
# Set the memory slurm parameter in the resources argument of plan(batchtools_slurm) to 4000L
# Before attempting to run this using tar_make_future()
targets::tar_make(perms_flynet_v1_studyforrest)
# Set the memory slurm parameter in the resources argument of plan(batchtools_slurm) to 32000L
# Before attempting to run this using tar_make_future()
# It isn't parallelized to run across multiple workers/slurm jobs so the one job needs to be large
targets::tar_make(statmaps_connectivity_studyforrest)
