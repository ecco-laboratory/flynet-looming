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
library(tibble)

# Set target options:
tar_option_set(
  packages = c("mixOmics",
               "tidymodels",
               "plsmod",
               "tidyverse",
               "magrittr",
               "glue",
               "withr",
               "matlabr",
               "rlang",
               "RNifti",
               "cowplot",
               "crayon"), # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
options(clustermq.scheduler = "slurm")
options(clustermq.template = "clustermq.tmpl")

# tar_make_future() configuration (okay to leave alone):
n_slurm_cpus <- 1L
plan(batchtools_slurm,
     template = "future.tmpl",
     resources = list(ntasks = 1L,
                      ncpus = n_slurm_cpus,
                      # nodelist = "node1",
                      # Must exclude gpu2 as R isn't updated on this node
                      # node3 is just slow and stinky for some reason
                      exclude = "gpu2,node3",
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 1800,
                      # The permutation tests for V1 are getting huge, they seem to need ~4 GB
                      # The Matlab-based permutation tests for model-based connectivity are also hefty
                      # they perform best with... 32 GB?! Just 2 b safe
                      memory = 4000L,
                      partition = "short"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 500
n_reps_per_batch <- 10

# Run the R scripts in the R/ folder with your custom functions:
tar_source(c("R/get_flynet_activation_timecourses.R",
             "R/get_retinotopy_fmri.R",
             "R/model_retinotopy_fmri.R",
             "R/plot_retinotopy_fmri.R",
             "R/utils/call-matlab.R",
             "R/utils/get-nifti.R",
             "R/utils/call-spm.R"))

matlab_path <- "/opt/MATLAB/R2024a/bin"

## data files from other people's stuff ----

# note that for this and any other objects that are read in from a different targets store,
# their updated-ness is only tracked by the original targets store
# so updates in these upstream will _not_ automatically render targets in this store out of date
# for our purposes I think it's okay because those utils should be stable, BUT for your edification,
# if there's any upstream things that you REALLY want tracked,
# best to re-declare them using tar_target() again
weights_flynet <- tar_read(weights_flynet, store = here::here("ignore", "_targets", "subjective"))

targets_stimuli <- list(
  tar_target(
    name = videos_studyforrest,
    # NOTE 2024-05-13!!! 
    # Original analyses were run using mp4 conversions of these original mkv videos
    # which are saved elsewhere
    # The flow extracted from the mkvs is EVER SO SLIGHTLY different (like seriously, correlation > .998)
    # so double check that all the results replicate with the new weights before you ship this to production
    command = list.files(here::here("ignore", 
                                    "datasets",
                                    "studyforrest-data-phase2",
                                    "code",
                                    "stimulus",
                                    "retinotopic_mapping",
                                    "videos"),
                         full.names = TRUE),
    format = "file"
  )
)

## matlab scripts (yuck) ----

targets_matlab <- list(
  tar_target(
    name = matlab_spmbatch_realign.norm,
    command = here::here("matlab", "preproc_fmri_data_spm.m"),
    format = "file"
  ),
  tar_target(
    name = matlab_preproc_mask_fmri_data_canlabtools,
    command = here::here("matlab", "preproc_mask_fmri_data_canlabtools.m"),
    format = "file"
  ),
  tar_target(
    name = matlab_calc_encoding_model_connectivity,
    command = here::here("matlab", "flynet_connectivity_studyforrest.m"),
    format = "file"
  )
)

## python scripts ----

py_calc_flynet_activations <- tar_read(py_calc_flynet_activations, store = here::here("ignore", "_targets", "subjective"))

## flynet activations ----

targets_flynet_activations <- list(
  tar_target(
    name = flynet_activations_raw_studyforrest,
    command = {
      video_paths <- paste(videos_studyforrest, collapse = " ")
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "flynet_activations.csv")
      
      system2(here::here("env", "bin", "python"),
              args = c(py_calc_flynet_activations,
                       "-l 132", # resized flow-frame height/width
                       paste("-i", video_paths),
                       paste("-o", out_path),
                       paste("-w", weights_flynet),
                       "-q activations"))
      
      out_path
      
      },
    format = "file"
  ),
  tar_target(
    name = flynet_activations_studyforrest,
    command = get_flynet_activation_studyforrest(flynet_activations_raw_studyforrest),
  ),
  tar_target(
    name = flynet_activations_preplot_studyforrest,
    command = flynet_activations_studyforrest %>% 
      pivot_longer(cols = starts_with("unit"), 
                   names_to = "unit_num", 
                   values_to = "activation", 
                   names_prefix = "unit_", 
                   names_transform = list(unit_num = as.integer)) %>% 
      mutate(unit_col = (unit_num-1) %% 16, 
             unit_row = (unit_num-1) %/% 16, 
             across(c(unit_col, unit_row), \(x) x - 7.5), 
             unit_ecc = sqrt(unit_col^2 + unit_row^2), 
             unit_angle = atan2(unit_row, unit_col))
  ),
  tar_target(
    name = flynet_activations_convolved_studyforrest,
    command = flynet_activations_studyforrest %>% 
      group_by(run_type) %>% 
      arrange(run_type, tr_num) %>% 
      mutate(across(starts_with("unit"), \(x) c(scale(convolve_hrf(x))))) %>% 
      ungroup()
  ),
  tar_target(
    name = flynet_activations_convolved_studyforrest_prematlab,
    command = {
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "flynet_convolved_timecourses.csv")
      
      flynet_activations_convolved_studyforrest %>% 
        select(stim_type = run_type, tr_num, everything()) %>% 
        # into the order that they will be in the fMRI data from matlab
        # alphabetical by studyforrest file condition name
        mutate(stim_type = fct_relevel(stim_type, "wedge_counter", "wedge_clock", "ring_contract")) %>% 
        arrange(stim_type, tr_num) %>% 
        write_csv(file = out_path)
      
      out_path
      },
    format = "file"
  )
)

## hand-modeled "activations" direct from looming variables ----

targets_formula_activations <- list(
  tar_target(
    formula_activations_studyforrest,
    # ring stimuli:
    # ring width 0.95 degrees of visual angle
    # 16 "timepoints," ring is a circle at time 0
    # so the outer radius at timestep t = 0 + 0.95*t
    command = crossing(run_type = c("ring_expand",
                                    "ring_contract",
                                    "wedge_clock",
                                    "wedge_counter"),
                       cycle_num = 1:5,
                       tr_num = 1:16) %>% 
      mutate(theta = case_when(
        run_type == "ring_expand" ~ 2*(0.95*pi/180)*tr_num,
        run_type == "ring_contract" ~ 2*(0.95*pi/180)*(17-tr_num),
        # we need it to be constant
        # so that d_theta will be 0
        # yeah it's a wedge so it's not a constant visual angle width
        # but this should give values that are roughly comparable with the rings
        TRUE ~ 0.95
      )) %>% 
      group_by(run_type) %>% 
      mutate(d_theta = c(0, diff(theta))) %>% 
      ungroup() %>% 
      mutate(tau_inv = d_theta / theta,
             eta = 1 * d_theta * exp(-1*theta),
             # +2 offset to align with the 2 empty TRs at the beginning of the run
             tr_num = (tr_num + (cycle_num-1)*16)+2) %>% 
      select(-cycle_num)
  ),
  tar_target(
    formula_activations_convolved_studyforrest,
    command = formula_activations_studyforrest %>% 
      select(run_type, tr_num, tau_inv, eta) %>% 
      arrange(run_type, tr_num) %>% 
      group_by(run_type) %>% 
      mutate(across(c(tau_inv, eta), \(x) c(scale(convolve_hrf(x)))),
             # because the wedge conditions come out as NaN after convolving
             # because they are constant, I suppose, lolz?
             across(c(tau_inv, eta), \(x) coalesce(x, 0))) %>% 
      ungroup()
  )
)

## fmri data input and preproc ----

valid_studyforrest_retinotopy_subjs <- list.files(here::here("ignore", "datasets", "studyforrest-data-phase2"), 
                                                  pattern=".*task-retmap.*run-1_bold.nii.gz", 
                                                  recursive = TRUE) %>% 
  stringr::str_split_i(pattern = "/", i = 1) %>% 
  unique()

targets_fmri_data <- list(
  tar_map(
    values = tibble(subject = valid_studyforrest_retinotopy_subjs),
    tar_target(
      bold_gz_wedge.clock,
      command = search_niftis_gz(subj_id = subject,
                                 task_id = "task-retmapclw"),
      format = "file"
    ),
    tar_target(
      bold_wedge.clock,
      command = gunzip_nifti(bold_gz_wedge.clock),
      format = "file"
    ),
    tar_target(
      bold_gz_wedge.counter,
      command = search_niftis_gz(subj_id = subject,
                                 task_id = "task-retmapccw"),
      format = "file"
    ),
    tar_target(
      bold_wedge.counter,
      command = gunzip_nifti(bold_gz_wedge.counter),
      format = "file"
    ),
    tar_target(
      bold_gz_ring.contract,
      command = search_niftis_gz(subj_id = subject,
                                 task_id = "task-retmapcon"),
      format = "file"
    ),
    tar_target(
      bold_ring.contract,
      command = gunzip_nifti(bold_gz_ring.contract),
      format = "file"
    ),
    tar_target(
      bold_gz_ring.expand,
      command = search_niftis_gz(subj_id = subject,
                                 task_id = "task-retmapexp"),
      format = "file"
    ),
    tar_target(
      bold_ring.expand,
      command = gunzip_nifti(bold_gz_ring.expand),
      format = "file"
    ),
    tar_target(
      bold_realign.norm,
      # these will get written into the studyforrest subject/ses directories
      # because SPM will not let you write preprocessed files into a directory other than
      # where the original niftis were saved
      command = spm_realign.norm_studyforrest(bold_wedge.counter,
                                              bold_wedge.clock,
                                              bold_ring.contract,
                                              bold_ring.expand,
                                     script = matlab_spmbatch_realign.norm),
      format = "file"
    )
  ),
  # TODO: get this target to aggregate across all bold_realign.norm
  tar_target(
    name = fmri_mat_sc_studyforrest,
    command = {
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "fmri_data_canlabtooled_sc.mat")

      # this actually does need the data to come in by subject
      matlab_commands <- c(
        rvec_to_matlabcell(
          list(
            bold_realign.norm_sub.01, 
            bold_realign.norm_sub.02, 
            bold_realign.norm_sub.03,
            bold_realign.norm_sub.04, 
            bold_realign.norm_sub.05, 
            bold_realign.norm_sub.06,
            bold_realign.norm_sub.09, 
            bold_realign.norm_sub.10, 
            bold_realign.norm_sub.14,
            bold_realign.norm_sub.15, 
            bold_realign.norm_sub.16, 
            bold_realign.norm_sub.17,
            bold_realign.norm_sub.18, 
            bold_realign.norm_sub.19, 
            bold_realign.norm_sub.20),
          matname = "all_files"),
        assign_variable("roi", "Bstem_SC"),
        assign_variable("out_name", out_path),
        call_script(matlab_preproc_mask_fmri_data_canlabtools)
      )
      
      with_path(
        matlab_path,
        run_matlab_code(matlab_commands)
      )
      
      out_path
      },
    format = "file"
  ),
  tar_target(
    name = fmri_mat_v1_studyforrest,
    command = {
      # the script now omits data from the bonky subject
      # as opposed to having one array dim filled with all 0s
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "fmri_data_canlabtooled_v1.mat")
      
      matlab_commands <- c(
        rvec_to_matlabcell(
          list(
            bold_realign.norm_sub.01, 
            bold_realign.norm_sub.02, 
            bold_realign.norm_sub.03,
            bold_realign.norm_sub.04, 
            bold_realign.norm_sub.05, 
            # skip subject 6--they had weird normalization? missing data in the V1 mask
            bold_realign.norm_sub.09, 
            bold_realign.norm_sub.10, 
            bold_realign.norm_sub.14,
            bold_realign.norm_sub.15, 
            bold_realign.norm_sub.16, 
            bold_realign.norm_sub.17,
            bold_realign.norm_sub.18, 
            bold_realign.norm_sub.19, 
            bold_realign.norm_sub.20),
          matname = "all_files"),
        assign_variable("roi", "Ctx_V1"),
        assign_variable("out_name", out_path),
        call_script(matlab_preproc_mask_fmri_data_canlabtools)
      )
      
      with_path(
        matlab_path,
        run_matlab_code(matlab_commands)
      )
      
      out_path
    },
    format = "file"
  ),
  tar_target(
    name = fmri_data_sc_studyforrest,
    command = fmri_mat_sc_studyforrest %>% 
      get_phil_matlab_fmri_data_studyforrest() %>% 
      proc_phil_matlab_fmri_data(tr_start = 3,
                                 tr_end = 82) %>% 
      mutate(stim_type = run_type)
  ),
  tar_target(
    name = groupavgs_fmri_sc_studyforrest,
    command = {
      fmri_data <- fmri_data_sc_studyforrest
      folds <- fmri_data %>% 
        prep_xval()
    
      folds %>% 
        mutate(groupavg = map(test_subjs,
                              \(x) fmri_data %>% 
                                filter(!(subj_num %in% x)) %>% 
                                calc_groupavg_timeseries(),
                              .progress = "Calculating group avg timeseries"))
    }
  ),
  tar_target(
    name = fmri_data_v1_studyforrest,
    command = fmri_mat_v1_studyforrest %>% 
      get_phil_matlab_fmri_data_studyforrest() %>% 
      proc_phil_matlab_fmri_data(tr_start = 3,
                                 tr_end = 82) %>% 
      mutate(stim_type = run_type)
      # Just sub-06 has all 0 data for some reason...
      # but fmri_mat_v1_studyforrest now omits 0-data subjects
      # so we should no longer need to filter them out post facto
  ),
  tar_target(
    name = groupavgs_fmri_v1_studyforrest,
    command = {
      fmri_data <- fmri_data_v1_studyforrest
      folds <- fmri_data %>% 
        prep_xval()
      
      folds %>% 
        mutate(groupavg = map(test_subjs,
                              \(x) fmri_data %>% 
                                filter(!(subj_num %in% x)) %>% 
                                calc_groupavg_timeseries(),
                              .progress = "Calculating group avg timeseries"))
    }
  )
)

## fmri model fitting for studyforrest SC data ----

targets_pls_sc_studyforrest <- list(
  tar_target(
    name = pls_flynet_sc_studyforrest,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_sc_studyforrest)
  ),
  tar_target(
    name = pls_pred.only_only.tauinv_sc_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest,
               predictor_cols = tau_inv,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_only.eta_sc_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest,
               predictor_cols = eta,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_only.combined_sc_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest,
               predictor_cols = c(tau_inv, eta),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_tauinv_sc_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                        by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest,
               predictor_cols = c(starts_with("unit"), tau_inv),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_eta_sc_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest,
               predictor_cols = c(starts_with("unit"), eta),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_combined_sc_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest,
               predictor_cols = c(starts_with("unit"), tau_inv, eta),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_flynet_sc_studyforrest,
    command = pls_flynet_sc_studyforrest %>% 
      select(-fits)
  ),
  tar_target(
    name = pls_flynet_sc_studyforrest_by.run.type,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_sc_studyforrest,
                       by_run_type = TRUE)
  ),
  tar_target(
    name = pls_pred.only_tauinv_sc_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(starts_with("unit"), tau_inv),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_eta_sc_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(starts_with("unit"), eta),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_combined_sc_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(starts_with("unit"), tau_inv, eta),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_only.tauinv_sc_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = tau_inv,
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_only.eta_sc_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = eta,
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_only.combined_sc_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_sc_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(tau_inv, eta),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_pred.only_flynet_sc_studyforrest_by.run.type,
    command = pls_flynet_sc_studyforrest_by.run.type %>% 
      select(-fits)
  )
)

## fmri model fitting for studyforrest V1 data ----

targets_pls_v1_studyforrest <- list(
  tar_target(
    name = pls_flynet_v1_studyforrest,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_v1_studyforrest,
                       include_fit = FALSE)
  ),
  tar_target(
    name = pls_only.tauinv_v1_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest,
               predictor_cols = tau_inv,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_only.eta_v1_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest,
               predictor_cols = eta,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_only.combined_v1_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest,
               predictor_cols = c(tau_inv, eta),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_tauinv_v1_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest,
               predictor_cols = c(starts_with("unit"), tau_inv),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_eta_v1_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest,
               predictor_cols = c(starts_with("unit"), eta),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_combined_v1_studyforrest,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest,
               predictor_cols = c(starts_with("unit"), tau_inv, eta),
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_flynet_v1_studyforrest_by.run.type,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_v1_studyforrest,
                       by_run_type = TRUE,
                       include_fit = FALSE)
  ),
  tar_target(
    name = pls_tauinv_v1_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(starts_with("unit"), tau_inv),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_eta_v1_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(starts_with("unit"), eta),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_combined_v1_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(starts_with("unit"), tau_inv, eta),
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_only.tauinv_v1_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = tau_inv,
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_only.eta_v1_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = eta,
               by_run_type = TRUE,
               include_fit = FALSE)
  ),
  tar_target(
    name = pls_only.combined_v1_studyforrest_by.run.type,
    command = flynet_activations_convolved_studyforrest %>% 
      left_join(formula_activations_convolved_studyforrest,
                by = c("run_type", "tr_num")) %>% 
      filter(startsWith(run_type, "ring")) %>% 
      fit_xval(in_y = fmri_data_v1_studyforrest %>% 
                 filter(startsWith(run_type, "ring")),
               predictor_cols = c(tau_inv, eta),
               by_run_type = TRUE,
               include_fit = FALSE)
  )
)

## model metrics ----

targets_metrics <- list(
  tar_target(
    name = metrics_flynet_sc_studyforrest,
    command = {
      pls_pred.only_flynet_sc_studyforrest %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_sc_studyforrest) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_flynet_sc_studyforrest_by.run.type,
    command = {
      pls_pred.only_flynet_sc_studyforrest_by.run.type %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_sc_studyforrest %>% 
                            unnest(groupavg) %>% 
                            nest(groupavg = -c(fold_num, run_type)),
                          decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_flynet_v1_studyforrest,
    command = {
      pls_flynet_v1_studyforrest %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_v1_studyforrest,
                          decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_flynet_v1_studyforrest_by.run.type,
    command = {
      pls_flynet_v1_studyforrest_by.run.type %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_v1_studyforrest %>% 
                            unnest(groupavg) %>% 
                            nest(groupavg = -c(fold_num, run_type)),
                          decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_formula_sc_studyforrest,
    command = bind_rows(tau_inv = pls_pred.only_tauinv_sc_studyforrest,
                        eta = pls_pred.only_eta_sc_studyforrest,
                        combined = pls_pred.only_combined_sc_studyforrest,
                        .id = "parameter") %>%
      wrap_pred_metrics(decoding = FALSE) %>% 
      select(-preds)
  ),
  tar_target(
    name = metrics_formula_sc_studyforrest_by.run.type,
    command = {
      bind_rows(tau_inv = pls_pred.only_tauinv_sc_studyforrest_by.run.type,
                eta = pls_pred.only_eta_sc_studyforrest_by.run.type,
                combined = pls_pred.only_combined_sc_studyforrest_by.run.type,
                .id = "parameter") %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_only.formula_sc_studyforrest,
    command = bind_rows(tau_inv = pls_pred.only_only.tauinv_sc_studyforrest,
                        eta = pls_pred.only_only.eta_sc_studyforrest,
                        combined = pls_pred.only_only.combined_sc_studyforrest,
                        .id = "parameter") %>%
      wrap_pred_metrics(decoding = FALSE) %>% 
      select(-preds)
  ),
  tar_target(
    name = metrics_only.formula_sc_studyforrest_by.run.type,
    command = {
      bind_rows(tau_inv = pls_pred.only_only.tauinv_sc_studyforrest_by.run.type,
                eta = pls_pred.only_only.eta_sc_studyforrest_by.run.type,
                combined = pls_pred.only_only.combined_sc_studyforrest_by.run.type,
                .id = "parameter") %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_formula_v1_studyforrest,
    command = bind_rows(tau_inv = pls_tauinv_v1_studyforrest,
                        eta = pls_eta_v1_studyforrest,
                        combined = pls_combined_v1_studyforrest,
                        .id = "parameter") %>%
      wrap_pred_metrics(decoding = FALSE) %>% 
      select(-preds)
  ),
  tar_target(
    name = metrics_formula_v1_studyforrest_by.run.type,
    command = {
      bind_rows(tau_inv = pls_tauinv_v1_studyforrest_by.run.type,
                eta = pls_eta_v1_studyforrest_by.run.type,
                combined = pls_combined_v1_studyforrest_by.run.type,
                .id = "parameter") %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_only.formula_v1_studyforrest,
    command = bind_rows(tau_inv = pls_only.tauinv_v1_studyforrest,
                        eta = pls_only.eta_v1_studyforrest,
                        combined = pls_only.combined_v1_studyforrest,
                        .id = "parameter") %>%
      wrap_pred_metrics(decoding = FALSE) %>% 
      select(-preds)
  ),
  tar_target(
    name = metrics_only.formula_v1_studyforrest_by.run.type,
    command = {
      bind_rows(tau_inv = pls_only.tauinv_v1_studyforrest_by.run.type,
                eta = pls_only.eta_v1_studyforrest_by.run.type,
                combined = pls_only.combined_v1_studyforrest_by.run.type,
                .id = "parameter") %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_all_studyforrest,
    command = {
      # SUMMARIZED ACROSS VOXELS!!! ALL OF THEM. IT WAS THE ONLY WAY
      sc_overall <- bind_rows(metrics_flynet_sc_studyforrest %>% 
                  mutate(parameter = "flynet"),
                metrics_only.formula_sc_studyforrest,
                metrics_formula_sc_studyforrest %>% 
                  mutate(parameter = paste0("flynet.", parameter))) %>% 
        select(parameter, perf) %>% 
        unnest(perf) %>% 
        group_by(parameter, stim_type, subj_num) %>% 
        summarize(r_model = mean(r_model))
      
      sc_by.run.type <- bind_rows(metrics_flynet_sc_studyforrest_by.run.type %>% 
                                mutate(parameter = "flynet"),
                              metrics_only.formula_sc_studyforrest_by.run.type,
                              metrics_formula_sc_studyforrest_by.run.type %>% 
                                mutate(parameter = paste0("flynet.", parameter))) %>% 
        select(parameter, perf) %>% 
        unnest(perf) %>% 
        group_by(parameter, stim_type, subj_num) %>% 
        summarize(r_model = mean(r_model))
      
      v1_overall <- bind_rows(metrics_flynet_v1_studyforrest %>% 
                                mutate(parameter = "flynet"),
                              metrics_only.formula_v1_studyforrest,
                              metrics_formula_v1_studyforrest %>% 
                                mutate(parameter = paste0("flynet.", parameter))) %>% 
        select(parameter, perf) %>% 
        unnest(perf) %>% 
        group_by(parameter, stim_type, subj_num) %>% 
        summarize(r_model = mean(r_model))
      
      v1_by.run.type <- bind_rows(metrics_flynet_v1_studyforrest_by.run.type %>% 
                                    mutate(parameter = "flynet"),
                                  metrics_only.formula_v1_studyforrest_by.run.type,
                                  metrics_formula_v1_studyforrest_by.run.type %>% 
                                    mutate(parameter = paste0("flynet.", parameter))) %>% 
        select(parameter, perf) %>% 
        unnest(perf) %>% 
        group_by(parameter, stim_type, subj_num) %>% 
        summarize(r_model = mean(r_model))
      
      sc <- left_join(sc_overall,
                      sc_by.run.type,
                      by = c("parameter", "stim_type", "subj_num"),
                      suffix = c("_overall", "_by.run.type"))
      
      v1 <- left_join(v1_overall,
                      v1_by.run.type,
                      by = c("parameter","stim_type", "subj_num"),
                      suffix = c("_overall", "_by.run.type")) %>% 
        # JUST IN CASE we might ever compare data between ROIs by subject
        # I think we never will because the noise ceilings are so different but just in case!
        # bc in the raw fMRI data coming in, subject 6 with the skunked voxels is just missing now
        mutate(subj_num = if_else(subj_num >= 6, subj_num + 1L, subj_num))
      
      bind_rows(SC = sc,
                V1 = v1,
                .id = "roi") %>% 
        ungroup()
  
    }
  )
)

## permute your life ----

targets_perms <- list(
  tar_rep(
    name = permuted_trs_studyforrest,
    command = {
      fmri_data <- fmri_data_sc_studyforrest
      tibble(permuted_trs = map(1,
                                \(x) get_permuted_order(fmri_data, 
                                                        n_cycles = 5L)))
      },
    batches = n_batches,
    reps = n_reps_per_batch,
    iteration = "list",
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep2(
    name = perms_flynet_sc_studyforrest,
    command = wrap_perms(permuted_trs = permuted_trs_studyforrest$permuted_trs,
                 preds_together = pls_pred.only_flynet_sc_studyforrest,
                 metrics_together = metrics_flynet_sc_studyforrest,
                 preds_by.run.type = pls_pred.only_flynet_sc_studyforrest_by.run.type,
                 metrics_by.run.type = metrics_flynet_sc_studyforrest_by.run.type),
    permuted_trs_studyforrest
  ),
  tar_rep2(
    name = perms_combined_sc_studyforrest,
    command = wrap_perms(permuted_trs = permuted_trs_studyforrest$permuted_trs,
                 preds_together = pls_pred.only_combined_sc_studyforrest,
                 metrics_together = metrics_formula_sc_studyforrest %>% 
                   filter(parameter == "combined"),
                 preds_by.run.type = pls_pred.only_combined_sc_studyforrest_by.run.type,
                 metrics_by.run.type =  metrics_formula_sc_studyforrest_by.run.type %>% 
                   filter(parameter == "combined")),
    permuted_trs_studyforrest
  ),
  # this set of perms keeps voxel num and only gets used to produce a statmap later
  # so it's not run on all the conditions so we can save time
  tar_rep2(
    name = perms_ring.expand_sc_studyforrest,
    command = pls_pred.only_flynet_sc_studyforrest_by.run.type %>% 
        rename(run_type = this_run_type) %>% 
        filter(run_type == "ring_expand") %>% 
        wrap_pred_metrics(permute_order = permuted_trs_studyforrest$permuted_trs,
                          decoding = FALSE) %>% 
        select(-preds) %>% 
        bind_perm_to_real(filter(metrics_flynet_sc_studyforrest_by.run.type, run_type == "ring_expand"), 
                          summarize_voxels = FALSE),
    permuted_trs_studyforrest
  ),
  tar_rep2(
    name = perms_flynet_v1_studyforrest,
    command = wrap_perms(permuted_trs = permuted_trs_studyforrest$permuted_trs,
                         preds_together = pls_flynet_v1_studyforrest,
                         metrics_together = metrics_flynet_v1_studyforrest,
                         preds_by.run.type = pls_flynet_v1_studyforrest_by.run.type,
                         metrics_by.run.type = metrics_flynet_v1_studyforrest_by.run.type),
    permuted_trs_studyforrest
  )
)

## other model outputs (in matlab from phil) ----

targets_metrics_matlab <- list(
  tar_target(
    name = statmaps_connectivity_studyforrest,
    command = {
      
      out_fstring <- here::here("ignore",
                                "outputs",
                                "retinotopy",
                                "%smap_flynet_connectivity_contrast.nii")
      out_paths <- sprintf(out_fstring, c("stat", "pval"))

      matlab_commands <- c(
        rvec_to_matlabcell(
          list(
            bold_realign.norm_sub.01, 
            bold_realign.norm_sub.02, 
            bold_realign.norm_sub.03,
            bold_realign.norm_sub.04, 
            bold_realign.norm_sub.05, 
            bold_realign.norm_sub.06,
            bold_realign.norm_sub.09, 
            bold_realign.norm_sub.10, 
            bold_realign.norm_sub.14,
            bold_realign.norm_sub.15, 
            bold_realign.norm_sub.16, 
            bold_realign.norm_sub.17,
            bold_realign.norm_sub.18, 
            bold_realign.norm_sub.19, 
            bold_realign.norm_sub.20),
          matname = "all_files"),
        assign_variable("sc_data_path", fmri_mat_sc_studyforrest),
        assign_variable("studyforrest_activation_path", flynet_activations_convolved_studyforrest_prematlab),
        assign_variable("out_fstring", out_fstring),
        call_script(matlab_calc_encoding_model_connectivity)
      )
      
      with_path(
        matlab_path,
        run_matlab_code(matlab_commands)
      )
      
      out_paths
    },
    format = "file"
  )
)

## tables? for supplements? ----

targets_tables <- list(
  tar_target(
    name = summary_metrics_all_studyforrest,
    command = {
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "supptable_retinotopy_model_perf.csv")
      metrics_all_studyforrest %>% 
        mutate(is_expand = if_else(stim_type == "ring_expand", 
                                   "Expanding rings", 
                                   "Other stimuli")) %>% 
        pivot_longer(cols = starts_with("r_model"), 
                     names_to = "fit_type", 
                     values_to = "r_model", 
                     names_prefix = "r_model_") %>% 
        group_by(roi, parameter, is_expand, fit_type) %>% 
        summarize(cv_r = mean(r_model), 
                  se = sd(r_model)/sqrt(n()),
                  .groups = "drop") %>% 
        filter(!is.na(cv_r)) %>% 
        mutate(across(where(is.numeric), \(x) signif(x, digits = 3)),
               parameter = fct_recode(parameter, 
                                      "Inverse tau" = "tau_inv",
                                      "Eta" = "eta",
                                      "Inverse tau and eta" = "combined",
                                      "Collision detection model" = "flynet",
                                      "Collision detection + inverse tau" = "flynet.tau_inv",
                                      "Collision detection + eta" = "flynet.eta",
                                      "Collision detection + inverse tau and eta" = "flynet.combined"),
               parameter = fct_relevel(parameter,
                                       "Inverse tau",
                                       "Eta",
                                       "Inverse tau and eta",
                                       "Collision detection model",
                                       "Collision detection + inverse tau",
                                       "Collision detection + eta",
                                       "Collision detection + inverse tau and eta"),
               fit_type = fct_recode(fit_type,
                                     "Stimulus-general" = "overall",
                                     "Stimulus-specific" = "by.run.type")) %>% 
        rename(ROI = roi,
               `Model predictors` = parameter,
               `Stimulus condition` = is_expand,
               `Model type` = fit_type,
               `Cross-validated r` = cv_r,
               `Standard error` = se
        ) %>% 
        write_csv(file = out_path)
      
      out_path
    },
    format = "file"
  )
)

## plotty plot plots ----

targets_plots <- list(
  tar_target(
    name = schematic_flynet_activation_timecourse,
    command = plot_flynet_activations_convolved(flynet_activations_convolved_studyforrest,
                                                run_types = "ring_expand")
  ),
  tar_target(
    name = boxplot_mini_cv.r_sc_studyforrest,
    command = plot_boxplot_mini_cv_r_studyforrest(metrics_all_studyforrest)
  ),
  tar_target(
    name = plot_intxn_cv.r_sc_studyforrest,
    command = plot_intxn_cv_r_studyforrest(metrics_all_studyforrest)
  ),
  tar_target(
    name = boxplot_mini_cv.r_v1_studyforrest,
    command = plot_boxplot_mini_cv_r_studyforrest(metrics_all_studyforrest, this_roi = "V1")
  ),
  tar_target(
    name = plot_intxn_cv.r_v1_studyforrest,
    command = plot_intxn_cv_r_studyforrest(metrics_all_studyforrest, this_roi = "V1")
  ),
  tar_target(
    name = heatmap_pls.beta_studyforrest_by.run.type,
    command = pls_flynet_sc_studyforrest_by.run.type %>% 
      select(fold_num, this_run_type, fits) %>% 
      mutate(mean_unit_betas = map(fits, \(x) x %>% 
                                     extract_fit_engine() %>% 
                                     pluck("mat.c") %>% 
                                     rowMeans())) %>% 
      select(-fits) %>% 
      unnest_longer(mean_unit_betas,
                    values_to = "mean_beta",
                    indices_to = "unit_num") %>% 
      separate(unit_num, into = c(NA, "unit_num"), convert = TRUE) %>% 
      mutate(unit_x = (unit_num-1) %% 16, unit_y = (unit_num-1) %/% 16) %>% 
      group_by(this_run_type, unit_num, unit_x, unit_y) %>% 
      summarize(mean_beta = mean(mean_beta), .groups = "drop") %>% 
      mutate(this_run_type = fct_recode(this_run_type, 
                                    "CW wedge" = "wedge_clock", 
                                    "CCW wedge" = "wedge_counter", 
                                    "Contracting ring" = "ring_contract", 
                                    "Expanding ring" = "ring_expand")) %>% 
      ggplot(aes(x = unit_x, y = -unit_y, fill = mean_beta)) + 
      geom_raster() + 
      scale_fill_viridis_c(option = "magma") + 
      facet_wrap(~ this_run_type) +
      labs(x = NULL,
           y = NULL,
           fill = "Encoding\nmodel beta")
  ),
  tar_target(
    name = schematic_rf_ecc,
    command = flynet_activations_preplot_studyforrest %>% 
      distinct(unit_num, unit_row, unit_col, unit_ecc) %>% 
      ggplot(aes(x = unit_col, y = unit_row, fill = unit_ecc)) + 
      geom_raster() + 
      scale_fill_viridis_c()
  ),
  tar_target(
    name = schematic_rf_angle,
    command = flynet_activations_preplot_studyforrest %>% 
      distinct(unit_num, unit_row, unit_col, unit_angle) %>% 
      ggplot(aes(x = unit_col, y = unit_row, fill = unit_angle)) + 
      geom_raster() + 
      scale_fill_viridis_c(option = "inferno")
  ),
  tar_target(
    name = schematic_flynet_activation_ecc,
    command = {
      color_range <- range(flynet_activations_preplot_studyforrest$unit_ecc)
      
      flynet_activations_preplot_studyforrest %>% 
        mutate(run_type = fct_recode(run_type, 
                                     "CW wedge" = "wedge_clock", 
                                     "CCW wedge" = "wedge_counter", 
                                     "Contracting ring" = "ring_contract", 
                                     "Expanding ring" = "ring_expand"),
               run_type = fct_relevel(run_type,
                                      "CW wedge",
                                      "CCW wedge",
                                      "Contracting ring",
                                      "Expanding ring")) %>% 
        nest(activations = -unit_num) %>% 
        slice_sample(prop = 0.2) %>% 
        unnest(activations) %>% 
        ggplot(aes(x = tr_num, y = activation, color = unit_ecc)) + 
        geom_line(aes(group = unit_num), alpha = 0.7) + 
        geom_vline(xintercept = (0:4)*16+3, linetype = "dotted") + 
        scale_color_viridis_c() + 
        expand_limits(color = color_range) +
        facet_wrap(~run_type)
    }
  ),
  tar_target(
    name = schematic_flynet_activation_angle,
    command = {
      color_range <- range(flynet_activations_preplot_studyforrest$unit_angle)
      
      flynet_activations_preplot_studyforrest %>% 
        mutate(run_type = fct_recode(run_type, 
                                     "CW wedge" = "wedge_clock", 
                                     "CCW wedge" = "wedge_counter", 
                                     "Contracting ring" = "ring_contract", 
                                     "Expanding ring" = "ring_expand"),
               run_type = fct_relevel(run_type,
                                      "CW wedge",
                                      "CCW wedge",
                                      "Contracting ring",
                                      "Expanding ring")) %>% 
        nest(activations = -unit_num) %>% 
        slice_sample(prop = 0.2) %>% 
        unnest(activations) %>% 
        ggplot(aes(x = tr_num, y = activation, color = unit_angle)) + 
        geom_line(aes(group = unit_num), alpha = 0.7) + 
        geom_vline(xintercept = (0:4)*16+3, linetype = "dotted") + 
        scale_color_viridis_c(option = "inferno") + 
        expand_limits(color = color_range) +
        facet_wrap(~run_type)
    }
  )
)

targets_figs <- list(
  tar_target(
    name = fig_schematic_flynet_activation,
    command = ggsave(here::here("ignore", "figs", "retinotopy_schematic_flynet_activation.svg"),
                     plot = schematic_flynet_activation_timecourse + 
                       guides(color = "none") + 
                       scale_color_manual(values = c("#c35413", "#41b6e6")) + 
                       labs(x = "Time", y = "Predicted BOLD") + 
                       theme_bw(base_size = 12) + 
                       theme(axis.text = element_blank(), plot.background = element_rect(fill = "transparent")),
                     width = 1200,
                     height = 600,
                     units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_boxplot_mini_cv.r_sc_studyforrest,
    command = plot_grid(boxplot_mini_cv.r_sc_studyforrest + 
                          theme_bw(base_size = 12) +
                          theme(plot.background = element_rect(fill = "transparent"),
                                axis.text.x = element_text(angle = 20, vjust = 0.7)), 
                        plot_intxn_cv.r_sc_studyforrest + 
                          scale_x_discrete(position = "top") + 
                          theme_bw(base_size = 12) +
                          theme(plot.background = element_rect(fill = "transparent"),
                                axis.text.x.top = element_text(angle = 20, vjust = 0.3)), 
                        rel_widths = c(2, 1), 
                        align = "h", 
                        axis = "tb") %>% 
      save_plot(filename = here::here("ignore", "figs", "retinotopy_boxplot_mini_cv.r_sc_studyforrest.png"),
                plot = .,
                base_asp = 1.8),
    format = "file"
  ),
  tar_target(
    name = fig_boxplot_mini_cv.r_v1_studyforrest,
    command = plot_grid(boxplot_mini_cv.r_v1_studyforrest + 
                          theme_bw(base_size = 12) +
                          theme(plot.background = element_rect(fill = "transparent"),
                                axis.text.x = element_text(angle = 20, vjust = 0.7)), 
                        plot_intxn_cv.r_v1_studyforrest + 
                          scale_x_discrete(position = "top") + 
                          theme_bw(base_size = 12) +
                          theme(plot.background = element_rect(fill = "transparent"),
                                axis.text.x.top = element_text(angle = 20, vjust = 0.3)), 
                        rel_widths = c(2, 1), 
                        align = "h", 
                        labels = "AUTO",
                        axis = "tb") %>% 
      save_plot(filename = here::here("ignore", "figs", "retinotopy_boxplot_mini_cv.r_v1_studyforrest.png"),
                plot = .,
                base_asp = 1.8),
    format = "file"
  ),
  tar_target(
    name = fig_heatmap_pls.beta_studyforrest_by.run.type,
    command = ggsave(here::here("ignore", "figs", "retinotopy_heatmap_pls.beta_studyforrest_by.run.type.png"),
                     plot = heatmap_pls.beta_studyforrest_by.run.type + 
                       theme_bw(base_size = 10) +
                       theme(legend.background = element_blank(),
                             plot.background = element_blank(),
                             axis.text = element_blank(),
                             aspect.ratio = 1),
                     width = 1400,
                     height = 1200,
                     units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_schematic_flynet_activation_ecc_angle,
    command = plot_grid(
      schematic_rf_ecc + 
        guides(fill = "none") + 
        labs(x = NULL, y = NULL) + 
        theme_bw(base_size = 12) + 
        theme(aspect.ratio = 1), 
      schematic_flynet_activation_ecc + 
        labs(x = "Time (TRs)", y = "Raw unit activation", color = "RF eccentricity\n(unit-widths)") + 
        theme_bw(base_size = 12), 
      schematic_rf_angle + 
        guides(fill = "none") + 
        labs(x = NULL, y = NULL) + 
        theme_bw(base_size = 12) + 
        theme(aspect.ratio = 1), 
      schematic_flynet_activation_angle + 
        labs(x = "Time (TRs)", y = "Raw unit activation", color = "RF polar angle\n(radians)") + 
        theme_bw(base_size = 12), 
      nrow = 2,
      labels = "AUTO", 
      rel_widths = c(1, 3)) %>% 
      save_plot(filename = here::here("ignore", "figs", "retinotopy_schematic_flynet_activation_ecc_angle.png"),
                plot = .,
                base_height = 6,
                base_asp = 2)
  )
)

## not so nifty niftis ----

targets_niftis <- list(
  tar_target(
    name = nifti.path_sc,
    # TODO: generate this from canlabtools script
    command = here::here("ignore", "utils", "retinotopy", "SC_mask_vox_indx.nii"),
    format = "file"
  ),
  tar_target(
    name = voxel.coords_sc,
    command = which(readNifti(nifti.path_sc) != 0, arr.ind = TRUE) %>% 
      as_tibble() %>% 
      rename(x = dim1, y = dim2, z = dim3) %>% 
      mutate(voxel_num = 1:n()) %>% 
      mutate(side = if_else(x <= 39, "right", "left"))
  ),
  tar_target(
    name = statmap_r_flynet_sc_ring.expand,
    command = {
      voxel_values <- perms_ring.expand_sc_studyforrest %>% 
        group_by(tar_batch, tar_rep, voxel_num) %>%
        summarize(r.model_real = mean(r.model_real),
                  r.model_perm = mean(r.model_perm),
                  .groups = "drop") %>% 
        left_join(voxel.coords_sc, by = "voxel_num") %>% 
        group_by(voxel_num, x, y, z, side) %>% 
        # fdr_correct expects a column called pval
        # write_statmap_nifti expects a column called r_model
        summarize(pval = (sum(r.model_perm > r.model_real)+1)/(n()+1), 
                  r_model = mean(r.model_real),
                  .groups = "drop") %>% 
        fdr_correct() %>% 
        filter(pval < q_cutoff)
      
      write_statmap_nifti(metric_voxels = voxel_values,
                          mask_path = nifti.path_sc,
                          out_path = here::here("ignore", "outputs", "retinotopy", "SC_pls_r_ring_expand_flynet_p005.nii"))
    },
    format = "file"
  )
)
## the list of all the target metanames ----

c(
  targets_matlab,
  targets_stimuli,
  targets_flynet_activations,
  targets_formula_activations,
  targets_fmri_data,
  targets_pls_sc_studyforrest,
  targets_pls_v1_studyforrest,
  targets_metrics,
  targets_perms,
  targets_metrics_matlab,
  targets_tables,
  targets_plots,
  targets_figs,
  targets_niftis
)
