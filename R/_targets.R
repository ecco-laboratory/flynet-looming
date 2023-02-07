# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline # nolint

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes)

# Set target options:
tar_option_set(
  packages = c("tidyverse",
               "magrittr"), # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
options(clustermq.scheduler = "slurm")
options(clustermq.template = "clustermq.tmpl")

# tar_make_future() configuration (okay to leave alone):
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.

# Run the R scripts in the R/ folder with your custom functions:
tar_source(c("R/get_flynet_activation_timecourses.R",
             "R/get_retinotopy_fmri.R"))
# source("other_functions.R") # Source other scripts as needed. # nolint

# Replace the target list below with your own:

target_flynet_activations_raw_studyforrest <- tar_map(
  values = tibble(filename = list.files(here::here("ignore",
                                                   "outputs",
                                                   "flynet_activations",
                                                   "132x132_stride8",
                                                   "studyforrest_retinotopy"))),
  tar_target(name = flynet_activation_raw_studyforrest,
             command = here::here("ignore",
                                  "outputs",
                                  "flynet_activations",
                                  "132x132_stride8",
                                  "studyforrest_retinotopy",
                                  filename))
)

target_flynet_activations_convolved_studyforrest <- tar_combine(
  name = flynet_activations_convolved_studyforrest,
  target_flynet_activations_raw_studyforrest,
  command = get_flynet_activation_studyforrest(vctrs::vec_c(!!!.x))
)

target_flynet_activations_raw_nsd <- tar_map(
  values = tibble(filename = list.files(here::here("ignore",
                                                   "outputs",
                                                   "flynet_activations",
                                                   "132x132_stride8",
                                                   "nsd_retinotopy"))),
  tar_target(name = flynet_activation_raw_nsd,
             command = here::here("ignore",
                                  "outputs",
                                  "flynet_activations",
                                  "132x132_stride8",
                                  "nsd_retinotopy",
                                  filename))
)

target_flynet_activations_convolved_nsd <- tar_combine(
  name = flynet_activations_convolved_nsd,
  target_flynet_activations_raw_nsd,
  command = get_flynet_activation_nsd(vctrs::vec_c(!!!.x))
)

target_fmri_mat_sc_studyforrest <- tar_target(
  name = fmri_mat_sc_studyforrest,
  command = "/home/data/eccolab/studyforrest-data-phase2/DATA_bpf.mat",
  format = "file"
)

target_fmri_mat_v1_studyforrest <- tar_target(
  name = fmri_mat_v1_studyforrest,
  command = "/home/data/eccolab/studyforrest-data-phase2/V1_DATA_bpf.mat",
  format = "file"
)

target_fmri_mat_sc_nsd <- tar_target(
  name = fmri_mat_sc_nsd,
  command = "/home/mthieu/nsd_retinotopy_bold_sc.mat",
  format = "file"
)

target_fmri_data_sc_studyforrest <- tar_target(
  name = fmri_data_sc_studyforrest,
  command = fmri_mat_sc_studyforrest %>% 
    get_phil_matlab_fmri_data_studyforrest() %>% 
    proc_phil_matlab_fmri_data(region = "sc",
                               tr_start = 3,
                               tr_end = 82)
)

target_fmri_data_v1_studyforrest <- tar_target(
  name = fmri_data_v1_studyforrest,
  command = fmri_mat_v1_studyforrest %>% 
    get_phil_matlab_fmri_data_studyforrest() %>% 
    proc_phil_matlab_fmri_data(region = "v1",
                               tr_start = 3,
                               tr_end = 82)
)

target_fmri_data_sc_nsd <- tar_target(
  name = fmri_data_sc_nsd,
  command = fmri_mat_sc_nsd %>% 
    get_phil_matlab_fmri_data_nsd() %>% 
    proc_phil_matlab_fmri_data(region = "sc",
                               tr_end = 301)
)

list(target_flynet_activations_raw_studyforrest,
     target_flynet_activations_convolved_studyforrest,
     target_flynet_activations_raw_nsd,
     target_flynet_activations_convolved_nsd,
     target_fmri_mat_sc_studyforrest,
     target_fmri_mat_v1_studyforrest,
     target_fmri_mat_sc_nsd,
     target_fmri_data_sc_studyforrest,
     target_fmri_data_v1_studyforrest,
     target_fmri_data_sc_nsd
     )
