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
                                            metrics_prf_sc) {
  out <- metrics_flynet_sc %>% 
    bind_rows("Collision detection model, all stimuli" = .,
              "Collision detection model, stim-specific" = metrics_flynet_sc_run,
              "Group-average pRF model, all stimuli" = metrics_prf_sc,
              .id = "model_type") %>% 
    select(model_type, perf) %>% 
    unnest(perf) %>% 
    group_by(model_type, stim_type, subj_num) %>% 
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
                                    "Group-average pRF model, all stimuli",
                                    "Collision detection model, all stimuli",
                                    "Collision detection model, stim-specific")) %>% 
    ggplot(aes(x = stim_type, y = cv_r, fill = model_type)) + 
    geom_hline(yintercept = 0, linetype = "dotted") + 
    # geom_hline(yintercept = 0.1, linetype = "dotted", color = "gray60") + 
    geom_boxplot(alpha = 0.8) + 
    # geom_jitter(alpha = 0.5, width = 0.1) + 
    scale_fill_manual(values = c("#348338", "#0033a0", "#f2a900")) +
    guides(x = guide_axis(angle = 30), color = "none") +
    labs(x = "Retinotopic stimulus type", y = "cross-validated r")
  
  return (out)
}
