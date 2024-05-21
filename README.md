# flynet-looming
code for paper: "Visual looming is a primitive for human emotion", Thieu, Ayzenberg, Lourenco, & Kragel (2024) iScience

DOCUMENTATION IN PROGRESS! Sorry about that... I am cleaning up the code! -Monica Thieu

## SHORT VERSION

**Follow these steps to install dependencies and run code.**

1. Clone the repo
1. Install the following:
    1. Matlab, ideally no older than R2022a
    1. datalad
    1. conda
    1. R 4.3.1 and RStudio
1. Edit the following:
    1. environment.yml: Set the env name and prefix to match the local path to your clone folder
    1. R/_targets_retinotopy.R: edit the `matlab_path` variable to point to your own Matlab install path
1. Run the following:
    1. In a terminal: `shell/init_project.sh`
    1. In RStudio (opened using the .Rproj file): `renv::restore()`, follow all prompts as necessary
    1. In the same RStudio, `source('run.R')` to **run the whole analysis pipeline.**

## LONG VERSION

### Pull code

First, clone this repository to your favorite location on your computer. :3 The setup scripts and analysis pipelines rely almost exclusively on setting things up in subfolders of the cloned project repo.

### Install/edit some dependencies manually

Next, there are a couple things you'll have to set up manually before you can start running my helper scripts.

#### Some other programs & command line tools

The pipeline requires the following programs and command line tools that I can't really auto-download for you. Please download the following if you don't have them already:

- **Matlab:** You just gotta have it, I'm sorry. The Study 1 fMRI analyses use SPM and SPM-dependent tools under the hood. The analyses were originally written on Matlab R2022a. It's probably not worth it to install R2022a--if you already have a newer version, use that, or if you need Matlab for something else install the newest version and use that to reproduce this.
- [**datalad:**](https://handbook.datalad.org/en/latest/intro/installation.html#) used for downloading the _studyforrest_ extension retinotopic mapping data underlying Study 1.
- [**conda:**](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) used for Python analyses (the core neural network model is implemented through PyTorch) and maintaining the Python package environment
- **[R](https://cran.r-project.org) (ideally 4.3.1) and [RStudio:](https://posit.co/download/rstudio-desktop/)** used for R analyses (all tabular data analysis, major statistics, and graphs across 3 studies)

#### Study 3 video database

The Cowen & Keltner (2017) video database used for **Study 3** isn't immediately publicly available, and I didn't want to direct-link past Alan Cowen's data access form, so you'll need to download those videos manually.

Once you fill out [Alan's Google form for data access](https://docs.google.com/forms/d/e/1FAIpQLScf9XVemSUWz6kUWySUdaQ5pxwqs8mugngrkBoLmX-3DMX1KA/viewform), you will be provided with links to download a zip file of his stimulus videos. I recommend right-clicking to open the download links in a new tab, because if you navigate away from the form page you will have to fill it out again to access those links.

Move and rename the video folder (it will probably download with the folder name `mp4_noname`) so that the videos live inside the following relative path, assuming you are currently in the `flynet-looming` cloned directory:

```bash
ignore/datasets/subjective/stimuli/raw/
```

The video ratings (aggregated across raters per video) and other associated files for Study 3 are set up to download automatically using `osfr` and similar through the R pipeline later.

#### Manually edit some file paths

I have done everything possible to have the setup scripts and analysis pipeline set up everything to relative path to specific subfolders of the project repo, so that you need to edit as few file paths as possible. However, there were still a couple paths I couldn't totally automate that you will need to manually change before you continue:

1. In `environment.yml`, set the env name and prefix to match the local path to your clone folder
1. In `R/_targets_retinotopy.R`, the analysis pipeline script for Study 1, edit the `matlab_path` variable to point to your own Matlab install path. This is the only study that uses Matlab.

### Run first package setup script

I have included a `shell/init_project.sh` script that _should_ download the following pipeline dependencies for you. If you want to know what this setup script is doing, read on!

#### Download additional Matlab libraries

- [SPM12](https://github.com/spm/spm12)
- [canlab/CanlabCore](https://github.com/canlab/CanlabCore)
- [canlab/Neuroimaging_Pattern_Masks](https://github.com/canlab/Neuroimaging_Pattern_Masks)

These are all distributed as folders of functions that have no installer beyond cloning the repo to your local and then adding the folder of functions to your Matlab path. The `shell/init_project.sh` setup script should `git clone` all of the libraries into the `ignore/libraries` subfolder for you. All of the Matlab scripts point to these local install paths, so they should then find the folders correctly without additional modification!

If you have one or more of these libraries already (most likely SPM12), you can change the setup script to use `ln -s` (on Mac or Linux) to symlink your existing SPM12 folder to appear under `ignore/libraries/spm12/` as well, instead of git cloning a duplicate copy of SPM12.

#### Set up Python packages (with conda)

We have included a conda `environment.yml` file that should automatically install Python 3.9.13 (the version we used) and the Python package dependencies necessary to reproduce our analyses. `shell/init_project.sh` calls 

```python
conda env create -f environment.yml
```

for you.

#### Download Study 1 data with datalad

**Study 1**'s data all comes from the _studyforrest-data-phase2_ dataset, which is available for download with `datalad`. Once you have `datalad` installed, `shell/init_project.sh` will `datalad clone` the dataset metadata and `datalad get` the minimal necessary files (the retinotopy fMRI scans and the stimulus videos) onto your local machine in the `ignore/datasets/` subfolder that the analysis code expects.

### Set up R packages (with renv)

We have included an R project file and `renv` package environment lockfiles that should automatically install the R package dependencies necessary to reproduce our analyses.

We used R 4.3.1--renv will not manage the associated R version for you, only the packages, so we recommend you make sure you have this version installed before recreating our repository. renv does not _require_ you to have the specific R version encoded in the lockfile in order to restore a package environment, but I _strongly recommend_ you use the specific R version, which should allow renv to pull pre-compiled binaries for all package dependencies. You can make it work with a slightly newer R version, but renv may attempt to install certain packages from C++ source, which can cause you headaches.

If you have a different (probably newer) version of R installed, follow [these instructions](https://jacobrprice.github.io/2019/09/19/Installing-multiple-parallel-R-versions.html) to set up multiple side-by-side R versions on your Mac.

Follow [these instructions](https://support.posit.co/hc/en-us/articles/200486138-Changing-R-versions-for-the-RStudio-Desktop-IDE) to choose the active R version. For Mac users, I recommend installing and using the RSwitch menu bar utility.

Once you have R 4.3.1 (or whatever newer R version if you want to gamble with fate):

1. Open the .Rproj file in this repo. Once RStudio is open to the project folder, the renv autoloader should run on startup to install the renv package if you don't have it, and then the renv environment will activate.
1. Call `renv::restore()` to recreate the package environment.
1. Follow prompts as necessary to install all packages in the environment.



### Run pipeline through R

I use the [targets](https://books.ropensci.org/targets/) package to manage dependency tracking for data inputs, processing scripts, and outputs. (For example, the pipeline uses the `osfr` R package to automatically download Study 2's data (originally collected by authors Vlad Ayzenberg and Stella Lourenco) off of OSF for you and save it into a standard subfolder of the project folder.)

I have declared a separate targets pipeline for each study (Study 1: retinotopy, Study 2: eyeblink, Study 3: subjective) in three respective `R/_targets_[THIS_STUDY].R` scripts.

You can run the `run.R` script as follows:

```bash
cd path/to/flynet-looming
Rscript -e 'source("run.R")'
```

`run.R` attempts to call `targets::tar_make()` to execute all of the analysis code for Studies 3, 2, and then 1. (I have them written in that order because I started analyzing Study 3 first, and the Study 1 fMRI analyses take the longest to run. LOL) Hopefully it'll just... work!

If you have slurm set up for your computing environment, I have set up the three `_targets_*.R` scripts to work with `targets::tar_make_future(workers=n)` to attempt to parallel the analysis operations across n slurm jobs. Please read the extra notes I've given in `run.R` if you want to attempt slurm, because I fear some of it is environment-specific. I've tried to document everything I can but I make no promises about `tar_make_future()` working for you out of the box!
