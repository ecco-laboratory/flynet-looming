## setup ----

require(tidyverse)
require(magrittr)

## figure 1: flynet hit prob timeseries ----
# skipping the Clery et al 2020 marmoset stimulus video
# because the stimuli don't hit the face
video_names <- c(
  "lourenco",
  "baseball",
  "bower"
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
  mutate(video = fct_recode(video,
                            "Video of baseball hitting camera" = "baseball",
                            "Looming shadow modeled on Bower, Broughton, & Moore (1971)" = "bower",
                            "Looming rabbit from Vagnoni, Lourenco, & Longo (2012)" = "lourenco")) %>% 
  ggplot(aes(x = frame_num, y = phit)) + 
  # geom_line() + # the actual data
  geom_smooth(color = "black") + # smoothed for public consumption
  facet_wrap(~ video, ncol = 1, scales = "free_x") +
  labs(x = "Time elapsed (frames, frame rates differ per video)",
       y = "Estimated likelihood of eventual collision") +
  theme_bw()

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

metrics_sc %>% 
  select(stim_type, fold_num, pred) %>% 
  unnest(pred) %>% 
  filter(split_type == "test") %>% 
  group_by(stim_type, subj_num) %>% 
  summarize(cv_r = cor(obs, flynet), .groups = "drop") %>% 
  # group_by(stim_type) %>% 
  # summarize(cv_r = mean(correlation), .groups = "drop") %>% 
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
  ggplot(aes(x = stim_type, y = cv_r, color = highlight_me)) + 
  geom_hline(yintercept = 0, linetype = "dotted") + 
  geom_hline(yintercept = 0.1, linetype = "dotted", color = "gray60") + 
  geom_boxplot(alpha = 0.8) + 
  geom_jitter(alpha = 0.5, width = 0.1) + 
  scale_color_manual(values = c("black", "coral")) +
  guides(x = guide_axis(angle = 30), color = "none") +
  labs(x = "Retinotopic stimulus type", y = "cross-validated r") +
  theme_bw(base_size = 14)
