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
  packages = c("withr",
               "osfr",
               "tidymodels",
               "poissonreg",
               "tidyverse",
               "magrittr",
               "rlang",
               "factoextra",
               "cowplot"), # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

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
                      memory = 1000L,
                      partition = "short"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 10
n_reps_per_batch <- 1000

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

osf_project_id <- "as4vm"
osf_local_download_path <- here::here("ignore", "datasets", "eyeblink")
dir.create(osf_local_download_path, showWarnings = FALSE)

targets_raw_data <- list(
  tar_target(
    name = blink.counts_ayzenberg2015_raw,
    command = {
      osf_retrieve_node(osf_project_id) %>% 
        osf_ls_files() %>% 
        filter(name == "eyeblink") %>% 
        osf_ls_files() %>% 
        filter(name == "rawdata") %>% 
        osf_download(path = osf_local_download_path)
      
      list.files(file.path(osf_local_download_path, "rawdata"),
                 full.names = TRUE)
      },
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
        separate(video, into = c("stimulus", NA, NA, NA, NA, "ttc"), sep = "_") %>% 
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
        # join on tau and eta estimates
        left_join(tau.eta_ayzenberg2015 %>% 
                    select(frame_num, ttc, tau_inv, eta),
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
                         "make_looming_video.py"),
    format = "file"
  )
)

## homemade looming videos made from images ----

targets_stimuli <- list(
  tar_target(
    name = images_ayzenberg2015,
    command = {
      osf_retrieve_node(osf_project_id) %>% 
        osf_ls_files() %>% 
        filter(name == "eyeblink") %>% 
        osf_ls_files() %>% 
        filter(name == "stimuli") %>% 
        osf_download(path = osf_local_download_path)
      
      list.files(file.path(osf_local_download_path, "stimuli"),
                 full.names = TRUE)
      },
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
          
          with_path(here::here("env", "bin"),
                    code = system2("python", args = c(py_make_looming_video,
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
                    )))
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
      video_paths <- paste(videos_ayzenberg2015, collapse = " ")
      out_path <- here::here("ignore",
                             "outputs",
                             "eyeblink",
                             "flynet_activations.csv") 
      
      with_path(here::here("env", "bin"),
                code = system2("python",
                               args = c(py_calc_flynet_activations,
                                        "-l 132",
                                        paste("-i", video_paths),
                                        paste("-o", out_path),
                                        paste("-w", weights_flynet),
                                        "-q activations")))
      out_path
    },
    format = "file"
  ),
  tar_target(
    name = hit.probs_ayzenberg2015_raw,
    command = {
      video_paths <- paste(videos_ayzenberg2015, collapse = " ")
      out_path <- here::here("ignore",
                             "outputs",
                             "eyeblink",
                             "flynet_hitprobs.csv")
      
      with_path(here::here("env", "bin"),
                code = system2("python",
                               args = c(py_calc_flynet_activations,
                                        "-l 132",
                                        paste("-i", video_paths),
                                        paste("-o", out_path),
                                        paste("-w", weights_flynet),
                                        "-q hit_probs")))

      out_path
    },
    format = "file"
  ),
  tar_target(
    name = tau.eta_ayzenberg2015,
    command = {
      # Since we seem to have to do these just from my simulated videos:
      # start 15 obj-widths away, arrive 1 obj-width away
      # starting obj width was simulated at 15 deg visual angle
      object_width_deg <- 15
      object_width_dist <- 15 * 2 * tan(object_width_deg/2 * pi/180)
      
      hit.probs_ayzenberg2015 %>% 
        distinct(ttc, frame_num) %>% 
        group_by(ttc) %>% 
        mutate(distance = 15 - (14/(max(frame_num)-1))*(frame_num-1),
               theta = 2*atan2(object_width_dist/2, distance)) %>% 
        group_by(ttc) %>% 
        mutate(d_theta = c(NA, diff(theta))) %>% 
        ungroup() %>% 
        mutate(tau = theta / d_theta,
               tau_inv = d_theta / theta,
               eta = 1 * d_theta * exp(-1*theta))
    }
  )
)

## model fitting ----

# Bear in mind that r-squared doesn't make sense for Poisson regression
# And 

targets_models <- list(
  tar_target(
    name = model_blink.by.hit_flynet,
    command = workflow_eyeblink(blink.counts_by_hit.prob,
                                pca_vars = "hit_prob",
                                return_fit = TRUE)
  ),
  tar_target(
    name = model_blink.by.hit_only.tau.inv,
    command = workflow_eyeblink(blink.counts_by_hit.prob,
                                pca_vars = "tau_inv",
                                return_fit = TRUE)
  ),
  tar_target(
    name = model_blink.by.hit_only.eta,
    command = workflow_eyeblink(blink.counts_by_hit.prob,
                                pca_vars = "eta",
                                return_fit = TRUE)
  ),
  tar_target(
    name = model_blink.by.hit_tau.inv,
    command = workflow_eyeblink(blink.counts_by_hit.prob,
                                pca_vars = c("hit_prob", "tau_inv"),
                                return_fit = TRUE)
  ),
  tar_target(
    name = model_blink.by.hit_eta,
    command = workflow_eyeblink(blink.counts_by_hit.prob,
                                pca_vars = c("hit_prob", "eta"),
                                return_fit = TRUE)
  ),
  tar_target(
    name = model_blink.by.hit_combined,
    command = workflow_eyeblink(blink.counts_by_hit.prob,
                                pca_vars = c("hit_prob", "tau_inv", "eta"),
                                return_fit = TRUE)
  ),
  tar_target(
    name = pre.auc_blink.by.hit,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      mutate(n_blinks = if_else(n_blinks >= 5, "high", "low"),
             n_blinks = factor(n_blinks)) %>% 
      select(ttc, hit_prob, n_blinks)
  ),
  tar_target(
    name = auc_blink.by.hit,
    command = {
      auc_overall <- pre.auc_blink.by.hit %>% 
        roc_auc(truth = n_blinks, hit_prob)
      
      auc_by_ttc <- pre.auc_blink.by.hit %>% 
        group_by(ttc) %>% 
        roc_auc(truth = n_blinks, hit_prob)
      
      bind_rows(auc_by_ttc, auc_overall)
    }
  )
)

## permuting ----

targets_perms <- list(
  tar_rep(
    name = perms_blink.by.hit_flynet,
    command = perm_eyeblink(blink.counts_by_hit.prob,
                             pca_vars = "hit_prob",
                             times = n_reps_per_batch),
    batches = n_batches,
    reps = 1,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_blink.by.hit_only.tau.inv,
    command = perm_eyeblink(blink.counts_by_hit.prob,
                            pca_vars = "tau_inv",
                            times = n_reps_per_batch),
    batches = n_batches,
    reps = 1,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_blink.by.hit_only.eta,
    command = perm_eyeblink(blink.counts_by_hit.prob,
                            pca_vars = "eta",
                            times = n_reps_per_batch),
    batches = n_batches,
    reps = 1,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_blink.by.hit_tau.inv,
    command = perm_eyeblink(blink.counts_by_hit.prob,
                            pca_vars = c("hit_prob", "tau_inv"),
                            times = n_reps_per_batch),
    batches = n_batches,
    reps = 1,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_blink.by.hit_eta,
    command = perm_eyeblink(blink.counts_by_hit.prob,
                            pca_vars = c("hit_prob", "eta"),
                            times = n_reps_per_batch),
    batches = n_batches,
    reps = 1,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_blink.by.hit_combined,
    command = perm_eyeblink(blink.counts_by_hit.prob,
                            pca_vars = c("hit_prob", "tau_inv", "eta"),
                            times = n_reps_per_batch),
    batches = n_batches,
    reps = 1,
    storage = "worker",
    retrieval = "worker"
  ),
  # for auc, because I just don't feel like permuting the logistic regression fit
  # we will stick to the old faithful aka permuting the true values
  # to compare with the same preds
  tar_target(
    name = perms_auc_overall,
    command = pre.auc_blink.by.hit %>% 
      permutations(n_blinks, times = n_batches * n_reps_per_batch) %>% 
      mutate(.metrics = map(splits, \(x) x %>% 
                              analysis() %>% 
                              roc_auc(truth = n_blinks, hit_prob),
                            .progress = "permuting overall blink by hit AUC")) %>%
      select(-splits) %>% 
      unnest(.metrics)
  ),
  tar_target(
    name = boots_auc_overall,
    command = pre.auc_blink.by.hit %>% 
      bootstraps(times = n_batches * n_reps_per_batch) %>% 
      mutate(.metrics = map(splits, \(x) x %>% 
                              analysis() %>% 
                              roc_auc(truth = n_blinks, hit_prob),
                            .progress = "bootstrapping overall blink by hit AUC")) %>%
      select(-splits) %>% 
      unnest(.metrics)
  ),
  tar_target(
    name = perms_auc_ttc,
    command = pre.auc_blink.by.hit %>% 
      nest(data = -ttc) %>% 
      mutate(perms = map(data, \(x) permutations(x, n_blinks, times = n_batches * n_reps_per_batch))) %>% 
      select(-data) %>% 
      unnest(perms) %>% 
      mutate(.metrics = map(splits, \(x) x %>% 
                              analysis() %>% 
                              roc_auc(truth = n_blinks, hit_prob),
                            .progress = "permuting blink by hit AUC by TTC")) %>%
      select(-splits) %>% 
      unnest(.metrics)
  )
)

targets_perm_results <- list(
  tar_target(
    name = perm.pvals_auc_overall,
    command = perms_auc_overall %>% 
      left_join(auc_blink.by.hit %>% 
                  filter(is.na(ttc)),
                by = c(".metric", ".estimator"),
                suffix = c("_perm", "_real")) %>% 
      summarize(auc = mean(.estimate_real),
                pval = (sum(.estimate_perm > .estimate_real) + 1) / (n() + 1))
  ),
  tar_target(
    # for this one we are permuting and then testing the RANK CORRELATION
    # against the "ideal" rank correlation, aka 34567 TTC
    # the true test statistic will be the actual rank correlation
    name = perm.pvals_auc_ttc,
    command = {
      tau_real <- auc_blink.by.hit %>% 
        filter(!is.na(ttc)) %>% 
        summarize(tau = cor(ttc, .estimate)) %>% 
        pull(tau)
      perms_auc_ttc %>% 
        group_by(id) %>% 
        summarize(tau_perm = cor(ttc, .estimate, method = "kendall")) %>% 
        mutate(tau_real = tau_real) %>% 
        summarize(tau = mean(tau_real),
                  # less than BECAUSE OBSERVED TAU IS NEGATIVE!!!
                  pval = (sum(tau_perm < tau_real) + 1) / (n() + 1))
      }
  )
)

## tabley wablies for supplement ----

targets_tables <- list(
  tar_target(
    name = summary_blink.by.hit,
    command = {
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "supptable_eyeblink_model_perf.csv")
      
      list(flynet = model_blink.by.hit_flynet,
           tauinv = model_blink.by.hit_only.tau.inv,
           eta = model_blink.by.hit_only.eta,
           flynet.tauinv = model_blink.by.hit_tau.inv,
           flynet.eta = model_blink.by.hit_eta,
           combined = model_blink.by.hit_combined) %>% 
        map(glance) %>% 
        bind_rows(.id = "model_type") %>% 
        mutate(model_type = fct_recode(model_type,
                                       "Inverse tau" = "tauinv",
                                       "Eta" = "eta",
                                       "Collision detection model" = "flynet",
                                       "Collision detection + inverse tau" = "flynet.tauinv",
                                       "Collision detection + eta" = "flynet.eta",
                                       "Collision detection + inverse tau and eta" = "combined"),
               AIC = round(AIC, digits = 0)) %>% 
        select(`Model predictors` = model_type, AIC) %>% 
        arrange(AIC) %>% 
        write_csv(file = out_path)
      
      out_path
    },
    format = "file"
  )
)

## figgy wiggies ----

# remember you can't set the fonts on RStudio Cloud!

targets_plots <- list(
  tar_target(
    name = plot_blink.by.time,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      # .033 to get it in seconds, not milliseconds
      mutate(time = (frame_num_shifted - max(frame_num_shifted)) * .033) %>% 
      ggplot(aes(x = time, y = n_blinks, color = factor(ttc))) + 
      geom_line(aes(group = ttc)) + 
      scale_color_viridis_d(option = "magma", end = 0.9) + 
      guides(color = guide_legend(override.aes = list(linewidth = 5))) + 
      labs(x = "Pre-collision time (s)",
           y = "# blinks (summed across Ps)",
           color = "Time-to-contact (s)") +
      theme_bw() +
      theme(legend.position = c(0, 1), 
            legend.justification = c(0, 1), 
            legend.background = element_blank()),
  ),
  tar_target(
    name = plot_hit.by.time,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      # .033 to get it in seconds, not milliseconds
      mutate(time = (frame_num_shifted - max(frame_num_shifted)) * .033) %>% 
      ggplot(aes(x = time, y = hit_prob, color = factor(ttc))) + 
      geom_line(aes(group = ttc)) + 
      scale_color_viridis_d(option = "magma", end = 0.9) + 
      guides(color = guide_legend(override.aes = list(linewidth = 5))) + 
      labs(x = "Pre-collision time (s)",
           y = "Estimated P(hit)",
           color = "Time-to-contact (s)") +
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
      guides(color = guide_legend(override.aes = list(size = 3))) +
      labs(x = "Framewise hit probability",
           y = "# blinks (summed across Ps)",
           color = "Time to contact (s)")
  ),
  tar_target(
    name = plot_auc_blink.by.hit,
    command = pre.auc_blink.by.hit %>% 
      group_by(ttc) %>% 
      yardstick::roc_curve(truth = n_blinks, hit_prob) %>% 
      autoplot() +
      scale_color_viridis_d(option = "magma", end = 0.9) +
      labs(x = "1 - Specificity",
           y = "Sensitivity", color = "TTC (s)") +
      guides(color = guide_legend(override.aes = list(linewidth = 3)))
  ),
  tar_target(
    name = plot_pca_loadings,
    command = blink.counts_by_hit.prob %>% 
      filter(frame_num_shifted >= 125) %>% 
      select(`Collision probability` = hit_prob, `Inverse tau` = tau_inv, `Eta` = eta) %>% 
      prcomp(center = TRUE, scale. = TRUE) %>% 
      fviz_pca_var() +
      expand_limits(x = 1.3) +
      labs(title = NULL)
  )
)

targets_figs <- list(
  tar_target(
    name = fig_schematic_hit.prob,
    command = (hit.probs_ayzenberg2015 %>% 
                 filter(ttc == 3, stimulus == "Butterfly2") %>% 
                 mutate(time = frame_num * .033) %>% 
                 ggplot(aes(x = time, y = hit_prob)) +
                 geom_line() +
                 geom_smooth(color = "#41b6e6") +
                 ylim(0, 1) +
                 labs(x = "Time (s)",
                      y = "Estimated P(hit)") +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank())) %>% 
      ggsave(filename = "eyeblink_butterfly2_hit.prob.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1200,
             height = 600,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_hit.by.time,
    command = (plot_hit.by.time +
                 theme_bw(base_size = 12) +
                 theme(legend.position = c(0, 1), 
                       legend.justification = c(0, 1), 
                       legend.background = element_blank(),
                       plot.background = element_blank())) %>% 
      ggsave(filename = "eyeblink_hit.by.time.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1800,
             height = 1200,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_blink.by.time,
    command = (plot_blink.by.time +
                 theme_bw(base_size = 12) +
                 theme(legend.position = c(0, 1), 
                       legend.justification = c(0, 1), 
                       legend.background = element_blank(),
                       plot.background = element_blank())) %>% 
      ggsave(filename = "eyeblink_blink.by.time.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1800,
             height = 1200,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_blink.hit.by.time,
    command = plot_grid(plot_blink.by.time, 
                        plot_hit.by.time +
                          guides(color = "none"), 
                        labels = "AUTO") %>% 
      save_plot(filename = here::here("ignore", "figs", "eyeblink_blink.hit.by.time.png"),
                plot = .,
                base_asp = 2.5)
  ),
  tar_target(
    name = fig_blink.by.hit,
    command = (plot_blink.by.hit +
                 theme_bw(base_size = 12) +
                 theme(legend.position = c(0, 1), 
                       legend.justification = c(0, 1),
                       legend.background = element_blank(),
                       plot.background = element_blank())) %>% 
      ggsave(filename = "eyeblink_blink.by.hit.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1800,
             height = 1200,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_auc_blink.by.hit,
    command = (plot_auc_blink.by.hit +
                 theme(text = element_text(size = 12),
                       legend.position = c(1, 0),
                       legend.justification = c(1, 0),
                       legend.background = element_blank(),
                       plot.background = element_blank())) %>% 
      ggsave(filename = "eyeblink_auc_blink.by.hit.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1000,
             height = 1000,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_pca_loadings,
    command = (plot_pca_loadings +
                 theme(text = element_text(size = 12),
                       plot.background = element_blank(),
                       aspect.ratio = 2/2.3)) %>% 
      ggsave(filename = "eyeblink_pca.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1400,
             height = 1200,
             units = "px"),
    format = "file"
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
  targets_perm_results,
  targets_tables,
  targets_plots,
  targets_figs
)
