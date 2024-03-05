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

plot_boxplot_mini_cv_r_studyforrest <- function (metrics_all, this_roi = "SC") {
  
  plot_data <- metrics_all %>% 
    filter(roi == this_roi, parameter == "flynet") %>% 
    mutate(is_expand = if_else(stim_type == "ring_expand", "Expanding rings", "Other stimuli")) %>% 
    pivot_longer(cols = starts_with("r_model"), 
                 names_to = "fit_type", values_to = "r_model", names_prefix = "r_model_") %>% 
    group_by(is_expand, fit_type, subj_num) %>% 
    summarize(r_model = mean(r_model), .groups = "drop") %>% 
    mutate(fit_type = fct_recode(fit_type, "Stimulus-general" = "overall", "Stimulus-specific" = "by.run.type"))
  
  plot_summaries <- plot_data %>%
    group_by(is_expand, fit_type) %>% 
    summarize(across(r_model, list(mean = mean, sd = sd, se = \(x) sd(x)/sqrt(length(x)))))
  
  out <- plot_data %>% 
    ggplot(aes(x = fit_type, y = r_model)) + 
    geom_line(aes(group = subj_num), color = "grey60") + 
    geom_jitter(width = 0.03, color = "grey60", size = 2, alpha = 0.7) +
    geom_errorbar(aes(y = NULL,
                      ymin = r_model_mean - 2*r_model_se,
                      ymax = r_model_mean + 2*r_model_se),
                  data = plot_summaries, width = 0) +
    geom_line(aes(group = 1, y = r_model_mean), 
              data = plot_summaries) +
    geom_point(aes(y = r_model_mean), 
               data = plot_summaries, size = 3) +
    geom_hline(yintercept = 0, linetype = "dotted") + 
    scale_color_viridis_c(direction = -1) +
    facet_grid(~ is_expand) +
    labs(x = "Model type", y = "Cross-validated r")

  return (out)
}

plot_intxn_cv_r_studyforrest <- function (metrics_all, this_roi = "SC") {
  preplot <- metrics_all %>% 
    filter(roi == this_roi, parameter == "flynet") %>% 
    mutate(r_diff = r_model_by.run.type - r_model_overall, 
           is_expand = if_else(stim_type == "ring_expand", "Expanding rings", "Other stimuli")) %>% 
    group_by(is_expand, subj_num) %>% 
    summarize(r_diff = mean(r_diff))
  
  out <- preplot %>% 
    summarize(mean_r_diff = mean(r_diff), 
              se_diff = sd(r_diff)/sqrt(n())) %>% 
    ggplot(aes(x = is_expand, y = mean_r_diff)) + 
    geom_line(aes(y = r_diff, group = subj_num), data = preplot, color = "grey60") + 
    geom_jitter(aes(y = r_diff), data = preplot, width = 0.02, color = "grey60", size = 2, alpha = 0.7) +
    geom_hline(yintercept = 0, linetype = "dotted") +
    geom_errorbar(aes(ymin = mean_r_diff - 2*se_diff, ymax = mean_r_diff + 2*se_diff), width = 0) +
    geom_point(size = 3) +
    labs(x = NULL, y = expression(r[specific] - r[general]))
  
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

