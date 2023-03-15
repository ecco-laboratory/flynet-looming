## setup ----

# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline # nolint

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes)
library(future)
library(future.callr)
library(future.batchtools)

# Set target options:
tar_option_set(
  packages = c("tidymodels",
               "plsmod",
               "tidyverse",
               "magrittr"), # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
options(clustermq.scheduler = "slurm")
options(clustermq.template = "clustermq.tmpl")

# tar_make_future() configuration (okay to leave alone):
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.

# plan(multicore)
# eventually... when I figure out why slurm is activating R 3.6.3
n_slurm_cpus <- 1L
plan(batchtools_slurm,
     template = "future.tmpl",
     resources = list(ntasks = 1L,
                      ncpus = n_slurm_cpus,
                      nodelist = "node1",
                      walltime = 86400L,
                      memory = 2000L,
                      partition = "day-long"))

# Run the R scripts in the R/ folder with your custom functions:
# UGH the modeling one has to come first so the fucking overloaded packages will load first
tar_source(c("R/model_retinotopy_fmri.R",
             "R/get_flynet_activation_timecourses.R",
             "R/get_retinotopy_fmri.R",
             "R/model_flynet_affect.R"
             ))

conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("select", "dplyr")
conflicted::conflict_prefer("map", "purrr")
# source("other_functions.R") # Source other scripts as needed. # nolint

## metadata files from other people's stuff ----

target_ck2017_ratings <- tar_target(
  name = ck2017_ratings,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/video_ratings.csv",
  format = "file"
)

target_ck2017_censored <- tar_target(
  name = ck2017_censored,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/censored_video_ids.csv",
  format = "file"
)

target_ck2017_kragel2019_train <- tar_target(
  name = ck2017_kragel2019_train,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/train_video_ids.csv",
  format = "file"
)

target_ck2017_kragel2019_test <- tar_target(
  name = ck2017_kragel2019_test,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/test_video_ids.csv",
  format = "file"
)

target_ck2017_kragel2019_classes <- tar_target(
  name = ck2017_kragel2019_classes,
  command = {
    censored <- read_csv(ck2017_censored)
    bind_rows(train = read_csv(ck2017_kragel2019_train),
              test = read_csv(ck2017_kragel2019_test),
              .id = "split") %>% 
      filter(!(emotion %in% c("Pride",
                              "Satisfaction",
                              "Sympathy",
                              "Anger",
                              "Admiration",
                              "Calmness",
                              "Relief",
                              "Awkwardness",
                              "Triumph",
                              "Nostalgia"))) %>% 
      mutate(censored = video %in% c(censored$less.bad, censored$very.bad))
    }
)

target_zhou2022_weights <- tar_map(
  values = tibble(filename = list.files(here::here("ignore",
                                                   "models",
                                                   "zhou2022"))),
  tar_target(name = zhou2022_weights,
             command = here::here("ignore",
                                  "models",
                                  "zhou2022",
                                  filename),
             format = "file")
)

## python scripts ----

target_py_flynet_utils <- tar_target(
  name = py_flynet_utils,
  command = here::here("python",
                       "myutils",
                       "flynet_utils.py"),
  format = "file"
)

target_py_convert_flynet_weights <- tar_target(
  name = py_convert_flynet_weights,
  command = here::here("python",
                       "myutils",
                       "convert_flynet_weights.py"),
  format = "file"
)

target_py_calc_flynet_activations <- tar_target(
  name = py_calc_flynet_activations,
  command = here::here("python",
                       "myutils",
                       "calc_flynet_activations.py"),
  format = "file"
)

## flynet setup stuff ----

target_flynet_weights <- tar_target(
  name = flynet_weights,
  command = {
    system2("python", args = c(py_convert_flynet_weights, "-u 256"))
    here::here("ignore",
               "models",
               "MegaFlyNet256.pt")
    },
  format = "file"
)

## flynet activations ----

target_flynet_activations_raw_ck2017 <- tar_target(
  name = flynet_activations_raw_ck2017,
  command = {
    flynet_weights
    system2("python",
            args = c(py_calc_flynet_activations,
                     "-l 132",
                     "-p /home/mthieu/Repos/CowenKeltner",
                     "-v videos_10fps",
                     "-m metadata",
                     "-q activations"))
    "/home/mthieu/Repos/CowenKeltner/metadata/flynet_132x132_stride8_activations.csv"
    },
  format = "file"
)

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
                                  filename),
             format = "file")
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
                                  filename),
             format = "file")
)

target_flynet_activations_convolved_nsd <- tar_combine(
  name = flynet_activations_convolved_nsd,
  target_flynet_activations_raw_nsd,
  command = get_flynet_activation_nsd(vctrs::vec_c(!!!.x))
)

## beh model fitting ----

target_flynet_activations_fit_ck2017 <- tar_target(
  name = flynet_activations_fit_ck2017,
  command = get_flynet_activation_ck2017(flynet_activations_raw_ck2017, ck2017_kragel2019_classes)
)

## fmri data input and preproc ----

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

target_prf_mat_sc_studyforrest <- tar_target(
  name = prf_mat_sc_studyforrest,
  command = "/home/data/eccolab/studyforrest-data-phase2/pred_sc_prf_xval.mat",
  format = "file"
)

target_prf_mat_v1_studyforrest <- tar_target(
  name = prf_mat_v1_studyforrest,
  command = "/home/data/eccolab/studyforrest-data-phase2/pred_v1_prf_xval.mat",
  format = "file"
)

target_fmri_data_sc_studyforrest <- tar_target(
  name = fmri_data_sc_studyforrest,
  command = fmri_mat_sc_studyforrest %>% 
    get_phil_matlab_fmri_data_studyforrest() %>% 
    proc_phil_matlab_fmri_data(tr_start = 3,
                               tr_end = 82) %>% 
    mutate(stim_type = run_type)
)

target_fmri_data_v1_studyforrest <- tar_target(
  name = fmri_data_v1_studyforrest,
  command = fmri_mat_v1_studyforrest %>% 
    get_phil_matlab_fmri_data_studyforrest() %>% 
    proc_phil_matlab_fmri_data(tr_start = 3,
                               tr_end = 82) %>% 
    mutate(stim_type = run_type) %>% 
    # Just this subject has all 0 data for some reason...
    filter(subj_num != 6) 
)

target_fmri_data_sc_nsd <- tar_target(
  name = fmri_data_sc_nsd,
  command = fmri_mat_sc_nsd %>% 
    get_phil_matlab_fmri_data_nsd() %>% 
    proc_phil_matlab_fmri_data(tr_end = 301) %>% 
    label_stim_types_nsd()
)

target_prf_data_sc_studyforrest <- tar_target(
  name = prf_data_sc_studyforrest,
  command = prf_mat_sc_studyforrest %>% 
    get_phil_matlab_fmri_data_studyforrest() %>% 
    proc_phil_matlab_fmri_data() %>% 
    rename(fold_num = subj_num) %>% 
    # since these were generated without the rest TRs
    mutate(tr_num = tr_num + 2L)
)

target_prf_data_v1_studyforrest <- tar_target(
  name = prf_data_v1_studyforrest,
  command = prf_mat_v1_studyforrest %>% 
    get_phil_matlab_fmri_data_studyforrest() %>% 
    proc_phil_matlab_fmri_data() %>% 
    rename(fold_num = subj_num) %>% 
    # since these were generated without the rest TRs
    mutate(tr_num = tr_num + 2L)
)

## fmri model fitting ----

target_pls_flynet_sc_studyforrest <- tar_target(
  name = pls_flynet_sc_studyforrest,
  command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                     in_y = fmri_data_sc_studyforrest)
)

target_pls_flynet_sc_nsd <- tar_target(
  name = pls_flynet_sc_nsd,
  command = fit_xval(in_x = flynet_activations_convolved_nsd,
                     in_y = fmri_data_sc_nsd)
)

target_metrics_flynet_sc_studyforrest <- tar_target(
  name = metrics_flynet_sc_studyforrest,
  command = {
    pls_flynet_sc_studyforrest %>% 
      wrap_pred_metrics(in_y = fmri_data_sc_studyforrest) %>% 
      select(-preds)
  }
)

target_metrics_flynet_sc_nsd <- tar_target(
  name = metrics_flynet_sc_nsd,
  command = pls_flynet_sc_nsd %>% 
    wrap_pred_metrics(in_y = fmri_data_sc_nsd) %>%
    select(-preds)
)

## permute your life ----

n_batches <- 50
n_reps_per_batch <- 10

target_perms_flynet_sc_studyforrest <- tar_rep(
  name = perms_flynet_sc_studyforrest,
  command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                     in_y = fmri_data_sc_studyforrest,
                     permute_params = list(n_cycles = 5L)) %>% 
    wrap_pred_metrics(in_y = fmri_data_sc_studyforrest) %>% 
    select(-preds),
  batches = n_batches,
  reps = n_reps_per_batch,
  storage = "worker",
  retrieval = "worker"
)

target_perms_flynet_sc_nsd <- tar_rep(
  name = perms_flynet_sc_nsd,
  command = fit_xval(in_x = flynet_activations_convolved_nsd,
                     in_y = fmri_data_sc_nsd,
                     permute_params = list(n_cycles = 1L)) %>% 
    wrap_pred_metrics(in_y = fmri_data_sc_nsd) %>% 
    select(-preds),
  batches = n_batches,
  reps = n_reps_per_batch,
  storage = "worker",
  retrieval = "worker"
)

## the list of all the target metanames ----

list(target_ck2017_ratings,
     target_ck2017_censored,
     target_ck2017_kragel2019_train,
     target_ck2017_kragel2019_test,
     target_ck2017_kragel2019_classes,
     target_zhou2022_weights,
     target_py_flynet_utils,
     target_py_convert_flynet_weights,
     target_py_calc_flynet_activations,
     target_flynet_weights,
     target_flynet_activations_raw_ck2017,
     target_flynet_activations_fit_ck2017,
     target_flynet_activations_raw_studyforrest,
     target_flynet_activations_convolved_studyforrest,
     target_flynet_activations_raw_nsd,
     target_flynet_activations_convolved_nsd,
     target_fmri_mat_sc_studyforrest,
     target_fmri_mat_v1_studyforrest,
     target_fmri_mat_sc_nsd,
     target_prf_mat_sc_studyforrest,
     target_prf_mat_v1_studyforrest,
     target_fmri_data_sc_studyforrest,
     target_fmri_data_v1_studyforrest,
     target_fmri_data_sc_nsd,
     target_prf_data_sc_studyforrest,
     target_prf_data_v1_studyforrest,
     target_pls_flynet_sc_studyforrest,
     target_pls_flynet_sc_nsd,
     target_metrics_flynet_sc_studyforrest,
     target_metrics_flynet_sc_nsd,
     target_perms_flynet_sc_studyforrest,
     target_perms_flynet_sc_nsd
     )
