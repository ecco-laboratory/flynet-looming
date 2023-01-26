## setup ----

require(tidyverse)
require(magrittr)

## figure 1: flynet hit prob timeseries ----
# skipping the Clery et al 2020 marmoset stimulus video
# because the stimuli don't hit the face
video_names <- c(
  "lourenco",
  "baseball"
)

hit_probs <- tibble(video = video_names) %>%
  mutate(data = map(video, \(x) {
    glue::glue("hit_probs_{x}.txt") %>% 
      here::here("ignore", "outputs", "r01_app", .) %>% 
      read_csv(col_names = "phit") %>% 
      mutate(frame_num = 1:nrow(.))
  })
  ) %>% 
  unnest(data)

hit_probs %>% 
  ggplot(aes(x = frame_num, y = phit)) + 
  geom_line() + 
  geom_smooth() +
  facet_wrap(~ video, ncol = 1, scales = "free_x")

## figure 2: FlyNet does ok on retinotopic ring expand in SC ----

metrics_sc <- read_rds(here::here("ignore", "outputs", "studyforrest_retinotopy_sc_pls_metrics.rds"))

metrics_sc %>% 
  select(stim_type, fold_num, perf_subj) %>% 
  unnest(perf_subj) %>% 
  filter(encoding_type == "flynet") %>% 
  group_by(stim_type, subj_num) %>% 
  summarize(q2 = mean(q2), .groups = "drop") %>% 
  # must separately relevel and recode because recode doesn't change level order
  mutate(highlight_me = stim_type == "ring_expand",
         stim_type = fct_relevel(stim_type, 
                                 "wedge_clock", 
                                 "wedge_counter", 
                                 "ring_contract", 
                                 "ring_expand"),
         stim_type = fct_recode(stim_type, 
                                "CW wedge" = "wedge_clock", 
                                "CCW wedge" = "wedge_counter", 
                                "Contracting ring" = "ring_contract", 
                                "Expanding ring" = "ring_expand")) %>% 
  ggplot(aes(x = stim_type, y = q2, color = highlight_me)) + 
  geom_hline(yintercept = 0, linetype = "dotted") + 
  geom_hline(yintercept = 0.1, linetype = "dotted", color = "gray60") + 
  geom_boxplot(alpha = 0.8) + 
  geom_jitter(alpha = 0.5, width = 0.1) + 
  scale_color_manual(values = c("black", "coral")) +
  guides(x = guide_axis(angle = 30), color = "none") +
  labs(x = "Retinotopic stimulus type", y = "cross-validated R-squared") +
  theme_bw(base_size = 14)
