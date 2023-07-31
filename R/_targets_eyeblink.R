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
library(rlang)

# Set target options:
tar_option_set(
  packages = c("tidymodels",
               "tidyverse",
               "magrittr",
               "rlang"), # packages that your targets need to run
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
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 1800L,
                      # Please be mindful this is not THAT much. No honker dfs pls
                      memory = 250L,
                      partition = "short"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 50
n_reps_per_batch <- 200

# Run the R scripts in the R/ folder with your custom functions:
tar_source(c("R/model_partial_r.R",
             "R/model_eyeblink.R"
))

# Run the R scripts in the R/ folder with your custom functions:
# tar_source(c())
# remove base conda from the python path so it only sees local conda env
Sys.setenv(PATH = stringr::str_remove_all(Sys.getenv("PATH"), "/opt/software/anaconda3/bin:"))

## data files from other people's stuff ----

weights_flynet <- tar_read(weights_flynet, store = here::here("ignore", "_targets", "subjective"))

targets_raw_data <- list(
  tar_target(
    name = blink.counts_ayzenberg2015_raw,
    command = list.files("/home/mthieu/Repos/lourenco_ilooming",
                         full.names = TRUE),
    format = "file"
  )
)

targets_cleaned_data <- list(
  tar_target(
    blink.counts_ayzenberg2015,
    command = tibble(filename = blink.counts_ayzenberg2015_raw) %>%
      mutate(data = map(filename, read_csv)) %>%
      unnest(data) %>% 
      rename(subj_num = SUBJ) %>%
      mutate(subj_num = as.integer(subj_num),
             # get it in seconds
             ttc = as.integer(str_sub(filename, start = -8, end = -5)) / 1000L) %>%
      select(-filename, -starts_with("...")) %>%
      filter(!(subj_num %in% c(26, 38, 53:54))) %>%
      pivot_longer(cols = -c(subj_num, ttc),
                   names_to = "frame_num",
                   values_to = "n_blinks",
                   values_ptypes = list(n_blinks = integer()),
                   names_transform = list(frame_num = as.integer)) %>% 
      filter(!is.na(n_blinks))
  ),
  tar_target(
    hit.probs_ayzenberg2015,
    command = hit.probs_ayzenberg2015_raw %>% 
        read_csv() %>% 
        rename(frame_num = frame) %>% 
        separate(video, into = c("stimulus", NA, NA, "ttc"), sep = "_") %>% 
        mutate(frame_num = as.integer(frame_num + 1),
               ttc = as.integer(str_sub(ttc, start = 5L, end = 5L)))
  ),
  tar_target(
    blink.counts_by_hit.prob,
    command = {
      blink.counts_ayzenberg2015 %>% 
        filter(!is.na(n_blinks)) %>% 
        # summarize blink counts across babies
        group_by(frame_num, ttc) %>% 
        summarize(n_blinks = sum(n_blinks), .groups = "drop") %>% 
        # summarize hit probs across stimuli
        left_join(hit.probs_ayzenberg2015 %>% 
                    group_by(frame_num, ttc) %>% 
                    summarize(hit_prob = mean(hit_prob), .groups = "drop"),
                  by = c("frame_num", "ttc")) %>% 
        group_by(ttc) %>% 
        # so they all end at frame 200
        mutate(frame_num_shifted = frame_num + (200 - max(frame_num))) %>% 
        ungroup()
    }
  )
)

## python scripts ----

py_calc_flynet_activations <- tar_read(py_calc_flynet_activations, store = here::here("ignore", "_targets", "subjective"))

targets_python <- list(
  tar_target(
    name = py_make_looming_video,
    command = here::here("python",
                         "myutils",
                         "make_looming_video.py"),
    format = "file"
  )
)

## homemade looming videos made from images ----

targets_stimuli <- list(
  tar_target(
    name = images_ayzenberg2015,
    command = list.files("~/Repos/lourenco_ilooming/stimuli/images",
                         full.names = TRUE),
    format = "file"
  ),
  tar_target(
    name = videos_ayzenberg2015,
    command = {
      out_path <- "~/Repos/lourenco_ilooming/stimuli/videos"
      for (img in images_ayzenberg2015) {
        cat("starting image:", crayon::green(img), fill = TRUE)
        for (loom_time in 3:7) {
          # print any of the associated stimuli that need to be tracked
          # run the python script
          cat("loomtime:", crayon::green(loom_time), fill = TRUE)
          system2("python", args = c(py_make_looming_video,
                                     paste("--infile", img),
                                     paste("--outpath", out_path),
                                     # Adult Ps sat 40 cm away from monitor
                                     # and I think it needs to come 4 obj-widths away
                                     paste("--diststart 15"),
                                     paste("--distend 1"),
                                     paste("--loomtime", loom_time),
                                     # visual angle should be 15.1 degrees?
                                     # Which is... kind of huge?
                                     paste("--objwidth 100"),
                                     # to get it as close to 1 frame per 33 ms as possible
                                     paste("--fps 29")
          )) 
        }
      }
      # print the path to the output videos
      list.files(out_path,
                 full.names = TRUE)
    },
    format = "file"
  )
)

targets_activations <- list(
  tar_target(
    name = activations_ayzenberg2015,
    command = {
      weights_flynet
      videos_ayzenberg2015
      system2("python",
              args = c(py_calc_flynet_activations,
                       "-l 132",
                       "-p /home/mthieu/Repos/lourenco_ilooming",
                       "-v stimuli/videos",
                       "-m .",
                       "-q activations"))
      "/home/mthieu/Repos/lourenco_ilooming/flynet_132x132_stride8_activations.csv"
    },
    format = "file"
  ),
  tar_target(
    name = hit.probs_ayzenberg2015_raw,
    command = {
      weights_flynet
      videos_ayzenberg2015
      system2("python",
              args = c(py_calc_flynet_activations,
                       "-l 132",
                       "-p /home/mthieu/Repos/lourenco_ilooming",
                       "-v stimuli/videos",
                       "-m .",
                       "-q hit_probs"))
      "/home/mthieu/Repos/lourenco_ilooming/flynet_132x132_stride8_hit_probs.csv"
    },
    format = "file"
  )
)

## model fitting ----

# Bear in mind that r-squared doesn't make sense for Poisson regression
# And 

targets_models <- list(
  tar_target(
    name = coefs_blink.by.hit,
    command = blink.counts_by_hit.prob %>% 
      premodel_eyeblink() %>% 
      glm(n_blinks ~ hit_prob + ttc, family = "poisson", data = .) %>% 
      tidy()
  )
)

## permuting ----

targets_perms <- list(
  tar_target(
    name = perms_hit.prob,
    command = blink.counts_by_hit.prob %>% 
      premodel_eyeblink() %>% 
      perm_eyeblink_hit.prob(times = n_batches * n_reps_per_batch) %>% 
      filter(term == "hit_prob")
  ),
  tar_target(
    name = perms_ttc,
    command = blink.counts_by_hit.prob %>% 
      premodel_eyeblink() %>% 
      perm_eyeblink_ttc(times = n_batches * n_reps_per_batch) %>% 
      filter(term == "ttc")
  )
)

## figgy wiggies ----

# remember you can't set the fonts on RStudio Cloud!

targets_figs <- list(
  tar_target(
    name = plot_blink.by.time,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      # .033 to get it in seconds, not milliseconds
      mutate(time = frame_num_shifted * .033) %>% 
      ggplot(aes(x = time, y = n_blinks, color = factor(ttc))) + 
      geom_line(aes(group = ttc)) + 
      scale_color_viridis_d(option = "magma") + 
      guides(color = guide_legend(override.aes = list(size = 5))) + 
      labs(x = "Time (s)",
           y = "# blinks (summed across Ps)",
           color = "Time to collision (s)") +
      theme_bw() +
      theme(legend.position = c(0, 1), 
            legend.justification = c(0, 1), 
            legend.background = element_blank()),
  ),
  tar_target(
    name = plot_blink.by.hit,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      ggplot(aes(x = hit_prob, y = n_blinks)) + 
      geom_point(aes(color = factor(ttc)), alpha = 0.5) + 
      geom_smooth(method = "glm", 
                  method.args = list(family = "poisson"), 
                  color = "black") +
      scale_color_viridis_d(option = "magma", end = 0.9) +
      labs(x = "Framewise hit probability",
           y = "# blinks (summed across Ps)",
           color = "Time to collision (s)") +
      theme_bw() +
      theme(legend.position = c(0, 1), 
            legend.justification = c(0, 1), 
            legend.background = element_blank())
  ),
  tar_target(
    name = plot_auc_blink.by.hit,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      mutate(blink_cat = if_else(n_blinks >= 5, "high", "low"),
             blink_cat = fct_relevel(blink_cat, "high")) %>% 
      group_by(ttc) %>% 
      yardstick::roc_curve(truth = blink_cat, hit_prob) %>% 
      autoplot() +
      scale_color_viridis_d(option = "magma", end = 0.9) +
      theme(legend.position = c(1, 0),
            legend.justification = c(1, 0),
            legend.background = element_blank())
  )
)

## the list of all the target metanames ----

c(
  targets_python,
  targets_stimuli,
  targets_raw_data,
  targets_cleaned_data,
  targets_activations,
  targets_models,
  targets_perms,
  targets_figs
)
