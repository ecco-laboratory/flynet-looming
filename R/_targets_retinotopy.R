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
               "RNifti",
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
                      nodelist = "gpu1",
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 86400L,
                      memory = 16000L,
                      partition = "day-long"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 50
n_reps_per_batch <- 20

# Run the R scripts in the R/ folder with your custom functions:
tar_source(c("R/get_flynet_activation_timecourses.R",
             "R/get_retinotopy_fmri.R",
             "R/model_retinotopy_fmri.R",
             "R/plot_retinotopy_fmri.R"
             ))

# source("other_functions.R") # Source other scripts as needed. # nolint

## python scripts ----

## flynet activations ----

targets_flynet_activations <- list(
  tar_target(
    name = flynet_activations_raw_studyforrest,
    command = list.files(here::here("ignore",
                                    "outputs",
                                    "flynet_activations",
                                    "132x132_stride8",
                                    "studyforrest_retinotopy"),
                         full.names = TRUE),
    format = "file"
  ),
  tar_target(
    name = flynet_activations_convolved_studyforrest,
    command = get_flynet_activation_studyforrest(flynet_activations_raw_studyforrest)
  ),
  tar_target(
    name = flynet_activations_raw_nsd,
    command = list.files(here::here("ignore",
                                    "outputs",
                                    "flynet_activations",
                                    "132x132_stride8",
                                    "nsd_retinotopy"),
                         full.names = TRUE),
    format = "file"
  ),
  tar_target(
    name = flynet_activations_convolved_nsd,
    command = get_flynet_activation_nsd(flynet_activations_raw_nsd)
  )
)

## fmri data input and preproc ----

targets_fmri_data <- list(
  tar_target(
    name = fmri_mat_sc_studyforrest,
    command = "/home/data/eccolab/studyforrest-data-phase2/DATA_bpf.mat",
    format = "file"
  ),
  tar_target(
    name = fmri_mat_v1_studyforrest,
    command = "/home/data/eccolab/studyforrest-data-phase2/V1_DATA_bpf.mat",
    format = "file"
  ),
  tar_target(
    name = fmri_mat_sc_nsd,
    command = "/home/mthieu/nsd_retinotopy_bold_sc.mat",
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
      mutate(stim_type = run_type) %>% 
      # Just this subject has all 0 data for some reason...
      filter(subj_num != 6) 
  ),
  tar_target(
    name = groupavgs_fmri_v1_studyforrest,
    command = {
      fmri_data <- fmri_data_v1_studyforrest
      folds <- fmri_data %>% 
        prep_xval() %>% 
        filter(fold_num != 6)
      
      folds %>% 
        mutate(groupavg = map(test_subjs,
                              \(x) fmri_data %>% 
                                filter(!(subj_num %in% x)) %>% 
                                calc_groupavg_timeseries(),
                              .progress = "Calculating group avg timeseries"))
    }
  ),
  tar_target(
    name = fmri_data_sc_nsd,
    command = fmri_mat_sc_nsd %>% 
      get_phil_matlab_fmri_data_nsd() %>% 
      proc_phil_matlab_fmri_data(tr_end = 301) %>% 
      label_stim_types_nsd()
  )
)

targets_prfs <- list(
  # 06/01/2023 changed over to using the circular group avg bc... sigh y'all
  tar_target(
    name = prf_mat_sc_studyforrest,
    command = "/home/data/eccolab/studyforrest-data-phase2/pred_sc_prf_groupavg.mat",
    format = "file"
  ),
  
  tar_target(
    name = prf_mat_v1_studyforrest,
    command = "/home/data/eccolab/studyforrest-data-phase2/pred_v1_prf_groupavg.mat",
    format = "file"
  ),
  tar_target(
    name = prf_data_sc_studyforrest,
    command = prf_mat_sc_studyforrest %>% 
      get_phil_matlab_fmri_data_studyforrest() %>% 
      proc_phil_matlab_fmri_data() %>% 
      # I think this is easiest so that the PLS predictors always have "unit" as the prefix
      rename_with(\(x) str_replace(x, "voxel", "unit")) %>% 
      # what comes out as being called subj_num is actually fold_num
      # but right now bc using 1 fold everyone,
      # just drop it
      select(-subj_num, -run_num) %>% 
      # since these were generated without the rest TRs
      mutate(tr_num = tr_num + 2L)
  ),
  tar_target(
    name = prf_data_v1_studyforrest,
    command = prf_mat_v1_studyforrest %>% 
      get_phil_matlab_fmri_data_studyforrest() %>% 
      proc_phil_matlab_fmri_data() %>% 
      rename_with(\(x) str_replace(x, "voxel", "unit")) %>% 
      select(-subj_num, -run_num) %>% 
      mutate(tr_num = tr_num + 2L)
  ) 
)

## fmri model fitting ----

targets_pls <- list(
  tar_target(
    name = pls_flynet_sc_studyforrest,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_sc_studyforrest)
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
    name = pls_pred.only_flynet_sc_studyforrest_by.run.type,
    command = pls_flynet_sc_studyforrest_by.run.type %>% 
      select(-fits)
  ),
  tar_target(
    name = pls_flynet_v1_studyforrest,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_v1_studyforrest,
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
    name = pls_flynet_sc_nsd,
    command = fit_xval(in_x = flynet_activations_convolved_nsd,
                       in_y = fmri_data_sc_nsd)
  ),
  tar_target(
    name = pls_prf_sc_studyforrest,
    command = fit_xval(in_x = prf_data_sc_studyforrest,
                       in_y = fmri_data_sc_studyforrest,
                       include_fit = FALSE)
  ),
  tar_target(
    name = pls_prf_v1_studyforrest,
    command = fit_xval(in_x = prf_data_v1_studyforrest,
                       in_y = fmri_data_v1_studyforrest,
                       include_fit = FALSE)
  )
)

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
        filter(fold_num != 6) %>% 
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
        filter(fold_num != 6) %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_v1_studyforrest %>% 
                            unnest(groupavg) %>% 
                            nest(groupavg = -c(fold_num, run_type)),
                          decoding = FALSE) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_flynet_sc_nsd,
    command = pls_flynet_sc_nsd %>% 
      select(-fits) %>% 
      wrap_pred_metrics(in_y = fmri_data_sc_nsd) %>%
      select(-preds)
  ),
  tar_target(
    name = metrics_prf_sc_studyforrest,
    command = {
      pls_prf_sc_studyforrest %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_sc_studyforrest) %>% 
        select(-preds)
    }
  ),
  tar_target(
    name = metrics_prf_v1_studyforrest,
    command = {
      pls_prf_v1_studyforrest %>% 
        # this subject is skunked for some reason
        filter(fold_num != 6) %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_v1_studyforrest,
                          decoding = FALSE) %>% 
        select(-preds)
    }
  )
)

## permute your life ----

targets_perms <- list(
  tar_rep(
    name = perms_flynet_sc_studyforrest,
    command = {
      permuted_trs <- get_permuted_order(fmri_data_sc_studyforrest, n_cycles = 5L)
      
      perms_together <- pls_pred.only_flynet_sc_studyforrest %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_sc_studyforrest,
                          permute_order = permuted_trs,
                          decoding = FALSE) %>% 
        select(-preds)
      
      perms_by_run_type <- pls_pred.only_flynet_sc_studyforrest_by.run.type %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(in_groupavg = groupavgs_fmri_sc_studyforrest %>% 
                            unnest(groupavg) %>% 
                            nest(groupavg = -c(fold_num, run_type)),
                          permute_order = permuted_trs,
                          decoding = FALSE) %>% 
        select(-preds)
      
      tibble(together = list(perms_together),
             by.run.type = list(perms_by_run_type))
    },
    batches = n_batches,
    reps = n_reps_per_batch,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_flynet_sc_nsd,
    command = pls_flynet_sc_nsd %>% 
      select(-fits) %>% 
      wrap_pred_metrics(in_y = fmri_data_sc_nsd,
                        permute_order = list(n_cycles = 1L)) %>% 
      select(-preds),
    batches = n_batches,
    reps = n_reps_per_batch,
    storage = "worker",
    retrieval = "worker"
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
    name = boxplot_cv.r_studyforrest,
    command = plot_boxplot_cv_r_studyforrest(metrics_flynet_sc_studyforrest,
                                             metrics_flynet_sc_studyforrest_by.run.type,
                                             metrics_prf_sc_studyforrest,
                                             metrics_flynet_v1_studyforrest,
                                             metrics_flynet_v1_studyforrest_by.run.type,
                                             metrics_prf_v1_studyforrest)
  )
)

targets_figs <- list(
  tar_target(
    name = fig_schematic_flynet_activation,
    command = ggsave(here::here("ignore", "figs", "ohbm2023_schematic_flynet_activation.png"),
                     plot = schematic_flynet_activation_timecourse + 
                       guides(color = "none") + 
                       scale_color_manual(values = c("#c35413", "#41b6e6")) + 
                       labs(x = "time", y = "predicted BOLD") + 
                       theme_bw(base_size = 18) + 
                       theme(axis.text = element_blank(), plot.background = element_rect(fill = "transparent")),
                     width = 6,
                     height = 4,
                     units = "in"),
    format = "file"
  ),
  tar_target(
    name = fig_boxplot_cv.r_studyforrest,
    command = ggsave(here::here("ignore", "figs", "ohbm2023_boxplot_cv.r_studyforrest.png"),
                     plot = boxplot_cv.r_studyforrest + 
                       theme_bw(base_size = 14) +
                       theme(legend.position = 0:1,
                             legend.justification = 0:1,
                             legend.background = element_blank(),
                             legend.title = element_blank(),
                             plot.background = element_rect(fill = "transparent")),
                     width = 8,
                     height = 6,
                     units = "in"),
    format = "file"
  )
)

targets_niftis <- list(
  tar_target(
    name = nifti.mask_sc,
    command = readNifti("/home/data/eccolab/studyforrest-data-phase2/SC_mask_vox_indx.nii")
  ),
  tar_target(
    name = voxel.coords_sc,
    command = which(nifti.mask_sc != 0, arr.ind = TRUE) %>% 
      as_tibble() %>% 
      rename(x = dim1, y = dim2, z = dim3) %>% 
      mutate(voxel_num = 1:n()) %>% 
      mutate(side = if_else(x <= 39, "right", "left"))
  ),
  tar_target(
    name = statmap_r_flynet_sc_ring.expand,
    command = {
      voxels <- metrics_flynet_sc_studyforrest_by.run.type %>% 
        filter(this_run_type == "ring_expand") %>% 
        preplot_voxels_model_r(voxel_coords = voxel.coords_sc)
      
      write_statmap_nifti(metric_voxels = voxels,
                          nifti_mask = nifti.mask_sc,
                          out_path = "/home/data/eccolab/studyforrest-data-phase2/SC_pls_r_ring_expand_flynet.nii")
    },
    format = "file"
  )
)
## the list of all the target metanames ----

c(
  targets_flynet_activations,
  targets_fmri_data,
  targets_prfs,
  targets_pls,
  targets_metrics,
  targets_perms,
  targets_plots,
  targets_figs,
  targets_niftis
)
