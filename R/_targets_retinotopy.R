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
               "dplyr",
               "forcats",
               "tidyr",
               "purrr",
               "ggplot2",
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
                      # nodelist = "node1",
                      exclude = "gpu2,node3",
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 86400L,
                      memory = 8000L,
                      partition = "day-long"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 100
n_reps_per_batch <- 50

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
                       in_y = fmri_data_v1_studyforrest %>% 
                         # re-number it to skip the missing subject 6
                         # so that the n folds thing works
                         mutate(subj_num = if_else(subj_num > 6, subj_num - 1L, subj_num)),
                       include_fit = FALSE)
  ),
  tar_target(
    name = pls_flynet_v1_studyforrest_by.run.type,
    command = fit_xval(in_x = flynet_activations_convolved_studyforrest,
                       in_y = fmri_data_v1_studyforrest %>% 
                         mutate(subj_num = if_else(subj_num > 6, subj_num - 1L, subj_num)),
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
        wrap_pred_metrics(permute_order = permuted_trs,
                          decoding = FALSE) %>% 
        select(-preds)
      
      perms_by_run_type <- pls_pred.only_flynet_sc_studyforrest_by.run.type %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(permute_order = permuted_trs,
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
    name = perms_flynet_v1_studyforrest,
    command = {
      permuted_trs <- get_permuted_order(fmri_data_v1_studyforrest, n_cycles = 5L)
      
      perms_together <- pls_flynet_v1_studyforrest %>% 
        wrap_pred_metrics(permute_order = permuted_trs,
                          decoding = FALSE) %>% 
        select(-preds) %>% 
        mutate(perf = map(perf, \(x) x %>% 
                            group_by(stim_type) %>% 
                            summarize(across(c(q2_model, r_model), mean))))
      
      perms_by_run_type <- pls_flynet_v1_studyforrest_by.run.type %>% 
        rename(run_type = this_run_type) %>% 
        wrap_pred_metrics(permute_order = permuted_trs,
                          decoding = FALSE) %>% 
        select(-preds) %>% 
        mutate(perf = map(perf, \(x) x %>% 
                            group_by(stim_type) %>% 
                            summarize(across(c(q2_model, r_model), mean))))
      
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

targets_perm_results <- list(
  tar_target(
    name = perm.pvals_flynet_sc_studyforrest,
    command = calc_perm_pvals(metrics_flynet_sc_studyforrest,
                              metrics_flynet_sc_studyforrest_by.run.type,
                              perms_flynet_sc_studyforrest)
  ),
  tar_target(
    name = perm.pvals_flynet_v1_studyforrest,
    command = calc_perm_pvals(metrics_flynet_v1_studyforrest,
                              metrics_flynet_v1_studyforrest_by.run.type,
                              perms_flynet_v1_studyforrest,
                              has_voxel_num = FALSE)
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
                                             metrics_flynet_v1_studyforrest,
                                             metrics_flynet_v1_studyforrest_by.run.type)
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
    name = fig_boxplot_cv.r_studyforrest,
    command = ggsave(here::here("ignore", "figs", "retinotopy_boxplot_cv.r_studyforrest.svg"),
                     plot = boxplot_cv.r_studyforrest + 
                       # previously was using "#348338" for the pRF
                       scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") +
                       theme_bw(base_size = 10) +
                       theme(legend.position = c(0,0),
                             legend.justification = c(0,0),
                             legend.background = element_blank(),
                             legend.title = element_blank(),
                             plot.background = element_rect(fill = "transparent")),
                     width = 1600,
                     height = 1200,
                     units = "px"),
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
  )
)

## not so nifty niftis ----

targets_niftis <- list(
  tar_target(
    name = nifti.path_sc,
    command = "/home/data/eccolab/studyforrest-data-phase2/SC_mask_vox_indx.nii",
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
      voxel_perms <- perms_flynet_sc_studyforrest %>% 
        select(tar_batch, tar_rep, by.run.type) %>% 
        unnest(by.run.type) %>% 
        select(-test_subjs) %>% 
        unnest(perf) %>% 
        filter(stim_type == "ring_expand") %>%
        group_by(tar_batch, tar_rep, voxel_num) %>%
        summarize(r_model_perm = mean(r_model),
                  .groups = "drop")
      
      voxel_values <- metrics_flynet_sc_studyforrest_by.run.type %>% 
        filter(run_type == "ring_expand") %>% 
        preplot_voxels_model_r(voxel_coords = voxel.coords_sc) %>% 
        right_join(voxel_perms, by = "voxel_num") %>% 
        group_by(voxel_num, x, y, z, side) %>% 
        summarize(pval = (sum(r_model_perm > r_model)+1)/(n()+1), 
                  r_model = mean(r_model),
                  .groups = "drop") %>% 
        fdr_correct() %>% 
        filter(pval < q_cutoff)
      
      write_statmap_nifti(metric_voxels = voxel_values,
                          mask_path = nifti.path_sc,
                          out_path = "/home/data/eccolab/studyforrest-data-phase2/SC_pls_r_ring_expand_flynet_p005.nii")
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
  targets_perm_results,
  targets_plots,
  targets_figs,
  targets_niftis
)
