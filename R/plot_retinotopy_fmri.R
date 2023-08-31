## setup ----
# Nothing to see here for now because this is targets-compatible

## plot specific functions ----

plot_flynet_activations_convolved <- function (activations, run_types = NULL) {
  
  if (!is.null(run_types)) {
    activations_filtered <- activations %>%
      filter(run_type %in% run_types)
  } else {
    activations_filtered <- activations
  }
  
  out <- activations_filtered %>% 
    pivot_longer(cols = starts_with("unit"), 
                 names_to = "unit_num", 
                 values_to = "activation", 
                 names_prefix = "unit_", 
                 names_transform = list(unit_num = as.integer)) %>% 
    filter(unit_num %in% c(128, 136)) %>% 
    ggplot(aes(x = tr_num, y = activation, 
               color = factor(unit_num))) + 
    geom_line(aes(group = factor(unit_num)))
  
  if (is.null(run_types) | length(run_types) > 1) {
    out <- out +
      facet_wrap(~ run_type, scales = "free_y")
  }
  return (out)
}

plot_boxplot_cv_r_studyforrest <- function (metrics_flynet_sc,
                                            metrics_flynet_sc_run,
                                            metrics_flynet_v1,
                                            metrics_flynet_v1_run) {
  out <- bind_rows("Collision detection model, all stimuli" = metrics_flynet_sc,
                   "Collision detection model, stim-specific" = metrics_flynet_sc_run,
                   # "Group-average pRF model, all stimuli" = metrics_prf_sc,
                   .id = "model_type") %>% 
    bind_rows(SC = .,
              V1 = bind_rows("Collision detection model, all stimuli" = metrics_flynet_v1,
                             "Collision detection model, stim-specific" = metrics_flynet_v1_run,
                             # "Group-average pRF model, all stimuli" = metrics_prf_v1,
                             .id = "model_type"),
              .id = "roi") %>% 
    select(roi, model_type, perf) %>% 
    unnest(perf) %>% 
    group_by(roi, model_type, stim_type, subj_num) %>% 
    summarize(cv_r = mean(r_model), .groups = "drop") %>% 
    # must separately relevel and recode because recode doesn't change level order
    mutate(stim_type = fct_relevel(stim_type, 
                                   "wedge_clock", 
                                   "wedge_counter", 
                                   "ring_contract", 
                                   "ring_expand"),
           stim_type = fct_recode(stim_type, 
                                  "CW wedge" = "wedge_clock", 
                                  "CCW wedge" = "wedge_counter", 
                                  "Contracting ring" = "ring_contract", 
                                  "Expanding ring" = "ring_expand"),
           model_type = fct_relevel(model_type,
                                    # "Group-average pRF model, all stimuli",
                                    "Collision detection model, all stimuli",
                                    "Collision detection model, stim-specific")) %>% 
    ggplot(aes(x = stim_type, y = cv_r, fill = model_type)) + 
    geom_hline(yintercept = 0, linetype = "dotted") + 
    # geom_hline(yintercept = 0.1, linetype = "dotted", color = "gray60") + 
    geom_boxplot(alpha = 0.8) + 
    # geom_jitter(alpha = 0.5, width = 0.1) + 
    facet_grid(roi ~ .) +
    guides(x = guide_axis(angle = 30), color = "none") +
    labs(x = "Retinotopic stimulus type", y = "Cross-validated r")
  
  return (out)
}

preplot_voxels_model_r <- function (metrics, voxel_coords) {
  out <- metrics %>% 
    select(fold_num, perf) %>%
    unnest(perf) %>% 
    group_by(voxel_num) %>% 
    summarize(r_model = mean(r_model)) %>% 
    left_join(voxel_coords, by = "voxel_num")
  
  return (out)
}

write_statmap_nifti <- function (metric_voxels, mask_path, out_path) {

  nifti_mask <- readNifti(mask_path)
  nifti_mask[ , , ] <- 0L
  # this is in glue pkg specification, seems easiest here
  # this_mask_path <- "/home/data/eccolab/studyforrest-data-phase2/SC_pls_r_ring_expand_flynet.nii"

  for (voxel in 1:nrow(metric_voxels)) {
    this_row <- metric_voxels %>% 
      slice(voxel)
      nifti_mask[this_row$x, this_row$y, this_row$z] <- this_row$r_model
  }
  
  RNifti::writeNifti(nifti_mask, out_path)
  return (out_path)
}
  
  if (FALSE) {
  # on the one that was run only on ring expand right now
  # because... time. pls
  sc_ring_expand_flynet_q2_suprathreshold_voxels <- metrics_sc_perm %>% 
    select(fold_num, iteration, perf_voxel) %>%
    unnest(perf_voxel) %>% 
    left_join(sc_ring_expand_flynet_q2_by_voxel %>% 
                group_by(fold_num, encoding_type, voxel_num) %>% 
                summarize(q2_obs = mean(q2_obs)),
              by = c("fold_num", "encoding_type", "voxel_num")) %>% 
    group_by(encoding_type, voxel_num) %>% 
    summarize(pval = mean(q2_obs < q2),
              q2_obs = mean(q2_obs)) %>% 
    arrange(pval) %>% 
    filter(pval < .05) %>% 
    left_join(mask_voxel_coords, by = "voxel_num")
  
  for (this_pval in c(.05, .01, .005)) {
    this_mask <- mask_nifti
    this_mask[ , , ] <- 0L
    # this is in glue pkg specification, seems easiest here
    this_mask_path <- "/home/data/eccolab/studyforrest-data-phase2/SC_pls_q2_ring_expand_flynet_p{str_sub(as.character(this_pval), start = 3L)}_mask.nii"
    these_voxels <- sc_ring_expand_flynet_q2_suprathreshold_voxels %>% 
      filter(pval < this_pval)
    for (voxel in 1:nrow(these_voxels)) {
      this_row <- these_voxels %>% 
        slice(voxel)
      this_mask[this_row$x, this_row$y, this_row$z] <- 1L
    }
    RNifti::writeNifti(this_mask, glue::glue(this_mask_path))
  }
  }

# a la Genovese, Lazar, & Nichols (2002)
fdr_correct <- function (metric_voxels, q = .05) {
  constant <- sum(1/1:length(metric_voxels))
  
  out <- metric_voxels %>% 
    arrange(pval) %>% 
    mutate(pval_num = 1:n(),
           q_cutoff = (pval_num/n()) * (q/constant))
  
  return (out)
}

