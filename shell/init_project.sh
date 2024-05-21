#!/bin/bash

# EDIT THIS TO MATCH YOUR LOCAL INSTALL PATH FOR OUR REPO!
REPO_PATH='/path/to/your/repo/install'
cd $REPO_PATH

# Make subdirectory structure matching my original setup so the relative paths work
mkdir ignore
mkdir ignore/datasets
mkdir ignore/figs
mkdir ignore/libraries
mkdir ignore/models
mkdir ignore/outputs
mkdir ignore/_targets
mkdir ignore/_targets/retinotopy
mkdir ignore/_targets/eyeblink
mkdir ignore/_targets/subjective
mkdir ignore/utils
mkdir ignore/utils/retinotopy

# Pull the matlab folders down from github into ignore/libraries
# If you already have any of these downloaded (say, SPM12)
# You can skip the git clone step and symlink your existing SPM12 folder with ln -s
# so that it shows up at ignore/libraries/spm12 instead
MATLAB_ADDON_PATH=${REPO_PATH}/ignore/libraries
cd $MATLAB_ADDON_PATH
git clone https://github.com/spm/spm12.git
git clone https://github.com/canlab/CanlabCore.git
git clone https://github.com/canlab/Neuroimaging_Pattern_Masks.git

# recreate conda env
conda env create -f environment.yml
# R package env is handled within R, so not in this setup script

# Pull Study 1 data with datalad (studyforrest retinotopy data)
datalad clone https://github.com/psychoinformatics-de/studyforrest-data-phase2 \
    ${REPO_PATH}/ignore/datasets

STUDYFORREST_PATH=${REPO_PATH}/ignore/datasets/studyforrest-data-phase2
cd $STUDYFORREST_PATH
# only download the bare minimum bc the data is large
datalad get code/stimulus/retinotopic_mapping/videos/
datalad get sub-*/ses-localizer/func/*retmap*bold.nii.gz
