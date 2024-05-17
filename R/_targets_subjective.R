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
library(osfr)

# Set target options:
tar_option_set(
  packages = c("osfr",
               "withr",
               "mixOmics",
               "tidymodels",
               "plsmod",
               "discrim",
               "tidyverse",
               "magrittr",
               "rlang",
               "cowplot"), # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
options(clustermq.scheduler = "slurm")
options(clustermq.template = "clustermq.tmpl")

# tar_make_future() configuration (okay to leave alone):
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.

# plan(multicore)
# eventually... when I figure out why slurm is activating R 3.6.3
n_slurm_cpus <- 1L
plan(batchtools_slurm,
     template = "future.tmpl",
     resources = list(ntasks = 1L,
                      ncpus = n_slurm_cpus,
                      # nodelist = "node1",
                      # exclude = "gpu2,node3",
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 1800L,
                      # Please be mindful this is not THAT much. No honker dfs pls
                      memory = 500L,
                      partition = "short"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 50
n_reps_per_batch <- 200

# Run the R scripts in the R/ folder with your custom functions:
tar_source(c("R/model_flynet_affect.R",
             "R/model_partial_r.R",
             "R/plot_helpers.R"
))


osf_project_id <- "as4vm"
osf_local_download_path <- here::here("ignore", "datasets", "subjective")
dir.create(osf_local_download_path, showWarnings = FALSE)
osf_folder <- osf_retrieve_node(osf_project_id) %>% 
    osf_ls_files() %>% 
    dplyr::filter(name == "subjective") %>% 
    osf_ls_files()

## metadata files from other people's stuff ----

targets_ratings <- list(
  tar_target(
    name = ratings_ck2017_raw,
    command = read_csv("https://s3-us-west-1.amazonaws.com/emogifs/CowenKeltnerEmotionalVideos.csv")
  ),
  tar_target(
    name = censored_ck2017,
    command = {
      download_path <- file.path(osf_local_download_path, "metadata")
      dir.create(download_path, showWarnings = FALSE)
      file_name <- "cowen2017_censored_video_ids.csv"
      
      osf_folder %>% 
        filter(name == "metadata") %>% 
        osf_ls_files() %>% 
        filter(name == file_name) %>% 
        osf_download(path = download_path,
                     conflicts = "overwrite")
      
      file.path(download_path, file_name)
      },
    format = "file"
  ),
  tar_target(
    name = ratings.loom_ck2017,
    command = {
      download_path <- file.path(osf_local_download_path, "rawdata")
      dir.create(download_path, showWarnings = FALSE)
      file_name <- "loom_ratings.csv"
      
      osf_folder %>% 
        filter(name == "rawdata") %>% 
        osf_ls_files() %>% 
        filter(name == file_name) %>% 
        osf_download(path = download_path,
                     conflicts = "overwrite")
      
      file.path(download_path, file_name)
    },
    format = "file"
  ),
  tar_target(
    name = ratings_ck2017,
    command = ratings_ck2017_raw %>% 
      full_join(read_csv(ratings.loom_ck2017),
                by = "Filename") %>% 
      rename(video = "Filename",
             looming = Looming,
             arousal = arousal...37)
  ),
  tar_target(
    name = ids.train_kragel2019,
    command = {
      download_path <- file.path(osf_local_download_path, "metadata")
      dir.create(download_path, showWarnings = FALSE)
      file_name <- "kragel2019_train_video_10fps_ids.csv"
      
      osf_folder %>% 
        filter(name == "metadata") %>% 
        osf_ls_files() %>% 
        filter(name == file_name) %>% 
        osf_download(path = download_path,
                     conflicts = "overwrite")
      
      file.path(download_path, file_name)
    },
    format = "file"
  ),
  tar_target(
    name = ids.test_kragel2019,
    command = {
      download_path <- file.path(osf_local_download_path, "metadata")
      dir.create(download_path, showWarnings = FALSE)
      file_name <- "kragel2019_test_video_10fps_ids.csv"
      
      osf_folder %>% 
        filter(name == "metadata") %>% 
        osf_ls_files() %>% 
        filter(name == file_name) %>% 
        osf_download(path = download_path,
                     conflicts = "overwrite")
      
      file.path(download_path, file_name)
    },
    format = "file"
  ),
  tar_target(
    name = classes_ck2017,
    command = {
      censored <- read_csv(censored_ck2017)
      bind_rows(train = read_csv(ids.train_kragel2019),
                test = read_csv(ids.test_kragel2019),
                .id = "split") %>% 
        filter(!(emotion %in% c("Pride",
                                "Satisfaction",
                                "Sympathy",
                                "Anger",
                                "Admiration",
                                "Calmness",
                                "Relief",
                                "Awkwardness",
                                "Triumph",
                                "Nostalgia"))) %>% 
        mutate(censored = video %in% c(censored$less.bad, censored$very.bad))
    }
  ),
  tar_target(
    name = classes_ck2017_csv,
    command = {
      out_path <- here::here("ignore",
                             "datasets",
                             "subjective",
                             "metadata",
                             "kragel2019_all_video_10fps_ids.csv")
      write_csv(classes_ck2017, file = out_path)
      
      out_path
      },
    format = "file"
  )
)

## stimuli!! ----

stimuli_path <- here::here("ignore",
                           "datasets",
                           "subjective",
                           "stimuli")

targets_stimuli <- list(
  tar_target(
    name = videos_cowen2017_raw,
    # NOTE TO USER!
    # You MUST make sure the videos you download _manually_ from Alan's form
    # end up at this path.
    # I don't feel comfortable pulling these from his AWS hosting
    # So this part is up to you, sadly
    command = list.files(file.path(stimuli_path, "raw"),
                         full.names = TRUE),
    format = "file"
  ),
  tar_target(
    name = videos_cowen2017_10fps,
    command = {
      videos_cowen2017_raw
      
      out_path <- file.path(stimuli_path, "fps10")
      dir.create(out_path,
                 showWarnings = FALSE)
      
      # manually adding env/bin to the shell search path with which this command will be called
      # is the caveman version of running this code using the local conda env
      # have to use the conda env to access the local install of ffmpeg
      with_path(here::here("env", "bin"),
                code = system2("python",
                               args = c(py_resample_video_fps,
                                        "-i", file.path(stimuli_path, "raw"),
                                        "-o", out_path))
      )
      
      list.files(out_path,
                 full.names = TRUE)
  
    },
    format = "file"
  )
)

## python scripts ----

targets_python <- list(
  tar_target(
    name = py_flynet_utils,
    command = here::here("python",
                         "myutils",
                         "flynet_utils.py"),
    format = "file"
  ),
  tar_target(
    name = py_convert_flynet_weights,
    command = {
      py_flynet_utils
      here::here("python",
                 "convert_flynet_weights.py")
    },
    format = "file"
  ),
  tar_target(
    name = py_calc_flynet_activations,
    command = {
      py_flynet_utils
      here::here("python",
                 "calc_flynet_activations.py")
    },
    format = "file"
  ),
  tar_target(
    name = py_resample_video_fps,
    command = here::here("python",
                         "resample_video_fps.py"),
    format = "file"
  ),
  tar_target(
    name = py_calc_emonet_activations,
    command = {
      py_flynet_utils
      here::here("python",
                 "calc_emonet_activations.py")
    },
    format = "file"
  )
)

## model weights ----

targets_weights <- list(
  tar_target(name = weights_zhou2022,
             command = {
               download_path <- here::here("ignore",
                                           "models")
               dir.create(download_path, showWarnings = FALSE)
               
               osf_retrieve_node(osf_project_id) %>% 
                 osf_ls_files() %>% 
                 filter(name == "weights") %>% 
                 osf_ls_files() %>% 
                 osf_download(path = download_path,
                              conflicts = "overwrite")
               
               list.files(download_path,
                          pattern = "zhou2022*.npy",
                          full.names = TRUE)
             },
             format = "file"
  ),
  tar_target(
    name = weights_flynet,
    command = {
      weights_zhou2022
      model_path <- here::here("ignore", "models")
      out_path <- file.path(model_path, "MegaFlyNet256.pt")
      system2("python", args = c(py_convert_flynet_weights, 
                                 "-u 256",
                                 paste("-i", model_path),
                                 paste("-o", out_path)))
      out_path
    },
    format = "file"
  ),
  tar_target(
    name = weights_emonet,
    command = {
      download_path <- here::here("ignore",
                                  "models")
      dir.create(download_path, showWarnings = FALSE)
      model_name <- "EmoNetPythonic.pt"
      # This is the Ecco Lab Model Zoo OSF repo
      osf_retrieve_node("pxvyb") %>% 
        osf_ls_files() %>% 
        dplyr::filter(name == model_name) %>% 
        osf_download(path = download_path,
                     conflicts = "overwrite")
      
      file.path(download_path, model_name)
    },
    format = "file"
  )
)

## emonet predictions on the videos ----

targets_preds_emonet <- list(
  tar_target(
    name = preds.framewise_emonet_ckvids,
    command = {
      videos_cowen2017_10fps
      out_path <- here::here("ignore",
                             "outputs",
                             "subjective",
                             "emonet_preds.csv") 
      
      with_path(here::here("env", "bin"),
                code = system2("python",
                               args = c(py_calc_emonet_activations,
                                        paste("-i", file.path(stimuli_path, "fps10")),
                                        paste("-o", out_path),
                                        paste("-w", weights_emonet),
                                        paste("-m", classes_ck2017_csv))))
      
      out_path
    },
    format = "file"
  ), 
  tar_target(
    name = preds.videowise_emonet_ckvids,
    command = {
      out <- read_csv(preds.framewise_emonet_ckvids) %>% 
        group_by(video) %>% 
        # amazingly, averaging each of the class probs across frames
        # yields class probs for each video that still add to 1. magical
        summarize(across(-frame, mean))
      
      emotion_classes <- out %>% 
        select(-video) %>% 
        colnames()
      
      out %>%
        rename_with(\(x) paste0(".pred_", x), .cols = -video) %>% 
        rowwise() %>% 
        # Do it rowwise to keep the class probs in their own cols while getting the max prob col
        mutate(emotion_pred = emotion_classes[c_across(starts_with(".pred")) == max(c_across(starts_with(".pred")))]) %>% 
        ungroup() %>% 
        left_join(read_csv(ids.test_kragel2019), by = "video") %>% 
        rename(emotion_obs = emotion) %>% 
        mutate(emotion_obs = factor(emotion_obs),
               # Pull the levels from the observed labels because empathic pain was never guessed
               emotion_pred = factor(emotion_pred, levels = levels(emotion_obs)))
    }
  )
)

## flynet activations ----

targets_activations_flynet <- list(
  tar_target(
    name = activations_flynet_ckvids,
    command = {
      weights_flynet
      
      video_paths <- videos_cowen2017_10fps
      out_fstring <- here::here("ignore",
                                "outputs",
                                "subjective",
                                "flynet_activations_%02d.csv") 
      n_batches <- 99
      batch_indices <- round(seq(1, n_batches, length.out = length(video_paths)))
      
      # break it up into batches bc the list-arg of videos can't handle that many characters
      for (batch in 1:n_batches) {
        out_path <- sprintf(out_fstring,
                            batch)
        these_videos <- paste(video_paths[batch_indices == batch], collapse = " ")
        
        system2("python",
                args = c(py_calc_flynet_activations,
                         "-l 132",
                         paste("-i", these_videos),
                         paste("-o", out_path),
                         paste("-w", weights_flynet),
                         "-q activations"))
      }
      
      sprintf(out_fstring, 1:n_batches)
    },
    format = "file"
  ),
  tar_target(
    name = rsplit_flynet_ckvids,
    command = {
      activations <- get_flynet_activation_ck2017(activations_flynet_ckvids, classes_ck2017) %>% 
        left_join(ratings_ck2017 %>% 
                    select(video, looming),
                  by = "video")
      # Use pre-existing Kragel 2019 train-test split to make an rsample-compatible split object
      make_splits(x = filter(activations, split == "train"),
                  assessment = filter(activations, split == "test"))
    }
  )
)

## beh model fitting ----

targets_preds_misc <- list(
  tar_target(
    name = preds_flynet_ckvids,
    command = {
      discrim_recipe <- rsplit_flynet_ckvids %>% 
        training() %>% 
        get_discrim_recipe()
      
      discrim_recipe %>% 
        get_discrim_workflow() %>% 
        fit(data = training(rsplit_flynet_ckvids)) %>% 
        get_discrim_preds_from_trained_model(in_recipe = discrim_recipe,
                                             test_data = testing(rsplit_flynet_ckvids))
    }
  ),
  tar_target(
    name = glm_looming.flynet_ckvids,
    command = {
      ratings <- ratings_ck2017
      
      this_recipe <- recipe(head(ratings)) %>% 
        update_role(arousal, valence, Fear, new_role = "predictor") %>% 
        update_role(looming, new_role = "outcome") %>% 
        update_role(video, new_role = "ID") %>% 
        step_normalize(all_predictors()) %>% 
        step_bin2factor(all_outcomes(), ref_first = FALSE, skip = TRUE)
      
      workflow() %>% 
        add_model(logistic_reg()) %>% 
        add_recipe(this_recipe) %>% 
        fit(data = ratings) %>%
        extract_fit_engine()
      }
  ),
  tar_target(
    name = recipe_looming.flynet_ckvids,
    command = rsplit_flynet_ckvids %>% 
        training() %>% 
        get_discrim_recipe(outcome_var = looming)
  ),
  tar_target(
    name = pls_looming.flynet_ckvids,
    command = recipe_looming.flynet_ckvids %>% 
        get_discrim_workflow(discrim_engine = pls(mode = "classification",
                                                  num_comp = 20L)) %>% 
        fit(data = training(rsplit_flynet_ckvids))
  ),
  tar_target(
    name = preds_looming.flynet_ckvids,
    command = pls_looming.flynet_ckvids %>% 
      get_discrim_preds_from_trained_model(in_recipe = recipe_looming.flynet_ckvids,
                                           test_data = testing(rsplit_flynet_ckvids),
                                           outcome_var = "looming")
  )
)

targets_model_confusions_bothnets_ckvids <- list(
  tar_target(
    name = coefs.confusions_bothnets_ckvids,
    command = get_confusion_regression_coefs(confusions_ckvids,
                                             lm_formulas = list(valence = diff_valence ~ dist_flynet + dist_emonet,
                                                                arousal = diff_arousal ~ dist_flynet + dist_emonet,
                                                                fear = diff_fear ~ dist_flynet + dist_emonet,
                                                                fear_no_arousal = diff_fear ~ dist_flynet + dist_emonet + diff_arousal,
                                                                looming = diff_looming ~ dist_flynet + dist_emonet))
    
  ),
  tar_target(
    name = partial.r2.confusions_bothnets_ckvids,
    command = {
      half_confusions <- confusions_ckvids %>% 
        halve_confusions()
      
      tribble(~outcome, ~term, ~partial.r2,
              "valence", "dist_flynet", calc_partial_r2(half_confusions, "diff_valence", "dist_flynet", "dist_emonet"),
              "valence", "dist_emonet", calc_partial_r2(half_confusions, "diff_valence", "dist_emonet", "dist_flynet"),
              "arousal", "dist_flynet", calc_partial_r2(half_confusions, "diff_arousal", "dist_flynet", "dist_emonet"),
              "arousal", "dist_emonet", calc_partial_r2(half_confusions, "diff_arousal", "dist_emonet", "dist_flynet"),
              "fear", "dist_flynet", calc_partial_r2(half_confusions, "diff_fear", "dist_flynet", "dist_emonet"),
              "fear", "dist_emonet", calc_partial_r2(half_confusions, "diff_fear", "dist_emonet", "dist_flynet"),
              "fear_no_arousal", "dist_flynet", calc_partial_r2(half_confusions, "diff_fear", "dist_flynet", c("dist_emonet", "diff_arousal")),
              "fear_no_arousal", "dist_emonet", calc_partial_r2(half_confusions, "diff_fear", "dist_emonet", c("dist_flynet", "diff_arousal")),
              "looming", "dist_flynet", calc_partial_r2(half_confusions, "diff_looming", "dist_flynet", "dist_emonet"),
              "looming", "dist_emonet", calc_partial_r2(half_confusions, "diff_looming", "dist_emonet", "dist_flynet")
      )
    }
  )
)

targets_stats_misc <- list(
  tar_target(
    name = cor.pred_ckvids,
    command = preds_flynet_ckvids %>% 
      bind_rows(flynet = .,
                emonet = preds.videowise_emonet_ckvids,
                .id = "model_type") %>% 
      select(model_type, video, emotion_obs, starts_with(".pred")) %>%
      pivot_longer(cols = starts_with(".pred"),
                   names_to = "class",
                   values_to = "prob") %>% 
      pivot_wider(names_from = model_type,
                  values_from = prob,
                  names_prefix = "prob_") %$% 
      cor(prob_flynet, prob_emonet)
  ),
  tar_target(
    name = pre.auc.by.category_bothnets_ckvids,
    command = bind_rows(flynet = preds_flynet_ckvids, 
                        emonet = preds.videowise_emonet_ckvids, 
                        .id = "model_type") %>% 
      pivot_longer(cols = starts_with(".pred"), 
                   names_to = "this_emotion", 
                   values_to = "classifier_prob", 
                   names_prefix = ".pred_") %>% 
      nest(probs = -c(model_type, this_emotion)) %>% 
      mutate(probs = map2(probs, this_emotion, 
                          \(x1, x2) mutate(x1, 
                                           across(c(emotion_obs, emotion_pred), 
                                                  \(y) fct_collapse(y, this_emotion = x2, other_level = "other")
                                           )
                          )
      )
      ) %>% 
      unnest(probs)
  ),
  tar_target(
    name = auc.by.category_bothnets_ckvids,
    command = pre.auc.by.category_bothnets_ckvids %>% 
      group_by(model_type, this_emotion) %>% 
      roc_auc(truth = emotion_obs, classifier_prob)
  )
)

## beh permutation testing ----

# I have returned to tar_rep...
# even though perm_beh_metrics() now takes finished predictions and only permutes the true outcomes
# leaving the predictions (and effectively the training model) the same every time
# models don't need to be refit, but AUC model accuracy takes a little while to estimate
# (1 min per 100 iterations, ish) thus we actually can really gain from paralleling again
# feed n_reps_per_batch directly into permutations()
# instead of the reps argument of tar_rep
# because we only want to call perm_beh_metrics once per iteration
# to minimize the number of times the constant ratings CSV gets read in

targets_perms <- list(
  tar_target(
    name = perms_looming.flynet_ckvids,
    command = preds_looming.flynet_ckvids %>% 
      permutations(permute = looming_obs, times = n_batches * n_reps_per_batch) %>% 
      mutate(.metrics = map(splits, \(x) x %>% 
                              analysis() %>% 
                              metrics(truth = looming_obs, 
                                      estimate = looming_pred, 
                                      .pred_yes),
                            .progress = "Bootstrapping Flynet to coded looming accuracy"
      )
      ) %>% 
      select(-splits)
  ),
  tar_target(
    name = boots_looming.flynet_ckvids,
    command = preds_looming.flynet_ckvids %>% 
      bootstraps(times = n_batches * n_reps_per_batch) %>% 
      mutate(.metrics = map(splits, \(x) x %>% 
                              analysis() %>% 
                              metrics(truth = looming_obs, 
                                      estimate = looming_pred, 
                                      .pred_yes),
                            .progress = "Bootstrapping Flynet to coded looming accuracy"
                            )
             ) %>% 
      select(-splits)
  ),
  tar_rep(
    name = perms_bothnets_ckvids,
    command = resample_beh_metrics(in_preds_flynet = preds_flynet_ckvids,
                                   in_preds_emonet = preds.videowise_emonet_ckvids,
                                   truth_col = emotion_obs,
                                   estimate_col = emotion_pred,
                                   pred_prefix = ".pred", 
                                   ratings = ratings_ck2017, 
                                   path_ids_train = ids.train_kragel2019,
                                   resample_type = "permute",
                                   times = n_reps_per_batch) %>% 
      select(-splits),
    batches = n_batches,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = boots_bothnets_ckvids,
    command = resample_beh_metrics(in_preds_flynet = preds_flynet_ckvids,
                                   in_preds_emonet = preds.videowise_emonet_ckvids,
                                   truth_col = emotion_obs,
                                   estimate_col = emotion_pred,
                                   pred_prefix = ".pred", 
                                   ratings = ratings_ck2017, 
                                   path_ids_train = ids.train_kragel2019,
                                   resample_type = "bootstrap",
                                   times = n_reps_per_batch) %>% 
      select(-splits),
    batches = n_batches,
    storage = "worker",
    retrieval = "worker"
  )
)
# doing this for 2x the usual number of permutations, with fewer batch forks,
# because the operation is fairly light so I want to minimize batch forks to keep the stuff trackable
target_perms.partial.r2_bothnets_ckvids <- tar_rep(
  name = perms.partial.r2_bothnets_ckvids,
  command = {
    valence <- confusions_ckvids %>% 
      halve_confusions() %>% 
      perm_partial_r2(y_col = "diff_valence",
                      x1_col = "dist_flynet",
                      x2_col = "dist_emonet",
                      times = n_reps_per_batch * 10)
    
    arousal <- confusions_ckvids %>% 
      halve_confusions() %>%
      perm_partial_r2(y_col = "diff_arousal",
                      x1_col = "dist_flynet",
                      x2_col = "dist_emonet",
                      times = n_reps_per_batch * 10)
    
    fear <- confusions_ckvids %>% 
      halve_confusions() %>%
      perm_partial_r2(y_col = "diff_fear",
                      x1_col = "dist_flynet",
                      x2_col = "dist_emonet",
                      times = n_reps_per_batch * 10)
    
    fear_no_arousal <- confusions_ckvids %>% 
      halve_confusions() %>%
      perm_partial_r2(y_col = "diff_fear",
                      x1_col = "dist_flynet",
                      x2_col = "dist_emonet",
                      covar_cols = "diff_arousal",
                      times = n_reps_per_batch * 10)
    
    looming <- confusions_ckvids %>% 
      halve_confusions() %>%
      perm_partial_r2(y_col = "diff_looming",
                      x1_col = "dist_flynet",
                      x2_col = "dist_emonet",
                      times = n_reps_per_batch * 10)
    
    bind_rows(valence = valence,
              arousal = arousal,
              fear = fear,
              fear_no_arousal = fear_no_arousal,
              looming = looming,
              .id = "outcome") %>% 
      rename(partial.r2_flynet = partial.r2_x1,
             partial.r2_emonet = partial.r2_x2) %>% 
      select(-splits)
  },
  batches = n_batches / 10,
  storage = "worker",
  retrieval = "worker"
)

targets_perm_misc <- list(
  tar_target(
    name = perms_cor.pred_ckvids,
    # Basically, shuffle the video labels for one of the models
    # bind them onto the true labels for the other model
    # pivot and correlate
    command = preds_flynet_ckvids %>% 
      left_join(preds.videowise_emonet_ckvids,
                by = c("video", "emotion_obs"),
                suffix = c("_flynet", "_emonet")) %>% 
      select(video, emotion_obs, starts_with(".pred")) %>% 
      permutations(ends_with("_flynet"), times = n_batches * n_reps_per_batch) %>% 
      mutate(correlation = map_dbl(splits,
                                   \(x) x %>% 
                                     analysis() %>% 
                                     # puts it longer by emotion category but keeps it wide by model type
                                     pivot_longer(cols = starts_with(".pred"),
                                                  names_to = c("class", ".value"),
                                                  names_sep = -6L) %$% 
                                     cor(flynet, emonet),
                                   .progress = "permuting correlation bw flynet/emonet preds")) %>% 
      select(-splits)
  ),
  tar_rep(
    name = boots_auc.by.category_bothnets_ckvids,
    command = left_join(preds_flynet_ckvids %>% 
                          select(-censored, -emotion_pred), 
                        preds.videowise_emonet_ckvids %>% 
                          select(-emotion_pred),
                        by = c("video", "emotion_obs"),
                        suffix = c("_flynet", "_emonet")) %>% 
      bootstraps(times = n_reps_per_batch * 5) %>% 
      mutate(.metrics = map(splits,
                            \(x) x %>% 
                              analysis() %>% 
                              pivot_longer(cols = starts_with(".pred"), 
                                           names_to = c(NA, "this_emotion", "model_type"), 
                                           values_to = "classifier_prob", 
                                           names_sep = "_") %>% 
                              nest(probs = -c(model_type, this_emotion)) %>% 
                              mutate(probs = map2(probs, this_emotion, 
                                                  \(x1, x2) mutate(x1, 
                                                                   emotion_obs = fct_collapse(emotion_obs, this_emotion = x2, other_level = "other")
                                                                   )
                                                  )
                              ) %>% 
                              unnest(probs) %>% 
                              group_by(model_type, this_emotion) %>% 
                              roc_auc(truth = emotion_obs, classifier_prob),
                            .progress = "bootstrapping AUC by emo category")) %>% 
      select(-splits),
    batches = n_batches / 5,
    storage = "worker",
    retrieval = "worker"
  ),
  tar_rep(
    name = perms_auc.by.category_bothnets_ckvids,
    command = left_join(preds_flynet_ckvids %>% 
                          select(-censored, -emotion_pred), 
                        preds.videowise_emonet_ckvids %>% 
                          select(-emotion_pred),
                        by = c("video", "emotion_obs"),
                        suffix = c("_flynet", "_emonet")) %>% 
      permutations(emotion_obs, times = n_reps_per_batch * 5) %>% 
      mutate(.metrics = map(splits,
                            \(x) x %>% 
                              analysis() %>% 
                              pivot_longer(cols = starts_with(".pred"), 
                                           names_to = c(NA, "this_emotion", "model_type"), 
                                           values_to = "classifier_prob", 
                                           names_sep = "_") %>% 
                              nest(probs = -c(model_type, this_emotion)) %>% 
                              mutate(probs = map2(probs, this_emotion, 
                                                  \(x1, x2) mutate(x1, 
                                                                   emotion_obs = fct_collapse(emotion_obs, this_emotion = x2, other_level = "other")
                                                  )
                              )
                              ) %>% 
                              unnest(probs) %>% 
                              group_by(model_type, this_emotion) %>% 
                              roc_auc(truth = emotion_obs, classifier_prob),
                            .progress = "permuting AUC by emo category")) %>% 
      select(-splits),
    batches = n_batches / 5,
    storage = "worker",
    retrieval = "worker"
  )
)

targets_perm_results <- list(
  tar_target(
    name = perm.pvals_model.acc_ckvids,
    command = {
      class_accuracies <- bind_rows(flynet = preds_flynet_ckvids,
                                    emonet = preds.videowise_emonet_ckvids,
                                    .id = "model_type") %>% 
        group_by(model_type) %>% 
        metrics(truth = emotion_obs,
                estimate = emotion_pred,
                starts_with(".pred"))
      
      perms_bothnets_ckvids %>% 
        select(starts_with(".metrics")) %>% 
        pivot_longer(starts_with(".metrics"),
                     names_to = "model_type",
                     values_to = ".metrics",
                     names_prefix = ".metrics_") %>% 
        unnest(.metrics) %>% 
        rename(.estimate_perm = .estimate) %>% 
        left_join(class_accuracies, by = c("model_type", ".metric", ".estimator")) %>% 
        group_by(model_type, .metric) %>% 
        # Carrying the accuracy value through so we can plot everything through this df alone
        summarize(accuracy = mean(.estimate),
                  pval = (sum(.estimate_perm >= .estimate) + 1) / (n() + 1)) %>% 
        mutate(pval_text = glue::glue("p = {signif(pval, digits = 2)}"))
    }
  ),
  tar_target(
    name = perm.pvals_auc.by.category_bothnets_ckvids,
    command = {
      tau_real <- auc.by.category_bothnets_ckvids %>% 
        pivot_wider(names_from = model_type, values_from = .estimate) %>%
        summarize(tau_real = cor(flynet, emonet, method = "kendall")) %>% 
        pull(tau_real)
      
      perms_auc.by.category_bothnets_ckvids %>% 
        unnest(.metrics) %>% 
        pivot_wider(names_from = model_type, values_from = .estimate) %>% 
        group_by(tar_batch, tar_rep, id) %>% 
        summarize(tau_perm = cor(flynet, emonet, method = "kendall"),
                  .groups = "drop") %>% 
        mutate(tau_real = tau_real) %>% 
        summarize(tau = mean(tau_real),
                  pval_upper = (sum(tau_perm > tau_real) + 1) / (n() + 1),
                  pval_lower = (sum(tau_perm < tau_real) + 1) / (n() + 1))
      }
  ),
  tar_target(
    name = perm.pvals_cor.pred_ckvids,
    command = perms_cor.pred_ckvids %>% 
      rename(correlation_perm = correlation) %>% 
      mutate(correlation_real = cor.pred_ckvids) %>% 
      summarize(correlation = mean(correlation_real),
                pval = (sum(correlation_perm > correlation_real) + 1) / (n() + 1))
  ),
  tar_target(
    name = perm.pvals_partial.r2_bothnets_ckvids,
    command = perms.partial.r2_bothnets_ckvids %>% 
      select(outcome, id, starts_with("partial.r2")) %>% 
      # because the real values are long by flynet/emonet
      pivot_longer(cols = starts_with("partial.r2"),
                   names_to = "model_type",
                   values_to = "partial.r2_perm",
                   names_prefix = "partial.r2_") %>% 
      left_join(partial.r2.confusions_bothnets_ckvids %>% 
                  rename(partial.r2_real = partial.r2) %>% 
                  mutate(term = str_sub(term, start = 6L)),
                by = c("outcome", "model_type" = "term")) %>% 
      group_by(outcome, model_type) %>% 
      summarize(partial.r2 = mean(partial.r2_real),
                pval = (sum(partial.r2_perm > partial.r2_real) + 1) / (n() + 1))
  )
)

## long-form confusions and distances ----

target_confusions_ckvids <- tar_target(
  name = confusions_ckvids,
  command = make_full_confusions(
    preds_flynet_ckvids,
    preds.videowise_emonet_ckvids,
    calc_distances_ratings(ratings_ck2017,
                           ids.train_kragel2019)
  )
)

target_mds.coords_ckvids <- tar_target(
  name = mds.coords_ckvids,
  command = {
    rating_means <- ratings_ck2017 %>% 
      select(video, arousal, valence) %>% 
      # Keep only the TRAINING videos
      # so this has the effect of "fitting" a "model" on the training videos
      inner_join(read_csv(ids.train_kragel2019), 
                 by = "video") %>% 
      group_by(emotion) %>% 
      summarize(across(c(arousal, valence), mean)) %>% 
      filter(emotion %in% levels(preds_flynet_ckvids$emotion_obs))
    
  list(emonet = convert_long_to_dist(distances = confusions_ckvids,
                                     row_col = emotion_obs,
                                     col_col = emotion_pred,
                                     y_col = dist_emonet,
                                     flip_dist = FALSE),
       flynet = convert_long_to_dist(distances = confusions_ckvids,
                                     row_col = emotion_obs,
                                     col_col = emotion_pred,
                                     y_col = dist_flynet,
                                     flip_dist = FALSE)) %>% 
    map(\(x) x %>% 
          MASS::isoMDS(y = rating_means %>% 
                         select(-emotion) %>% 
                         as.matrix() %>% 
                         scale()) %>% 
          pluck("points") %>% 
          set_colnames(c("x", "y")) %>% 
          as_tibble(rownames = "emotion")) %>%
    bind_rows(.id = "model_type") %>% 
    left_join(rating_means, by = "emotion")
  }
)

## tables for supplement sigh ----

targets_tables <- list(
  tar_target(
    name = summary_partial.r2_bothnets_ckvids,
    command = {
      out_path <- here::here("ignore",
                             "outputs",
                             "retinotopy",
                             "supptable_subjective_distance.model_perf.csv")
      
      perm.pvals_partial.r2_bothnets_ckvids %>% 
        ungroup() %>% 
        filter(!(outcome %in% c("fear_no_arousal", "looming"))) %>% 
        mutate(outcome = str_to_title(outcome),
               outcome = fct_relevel(outcome, "Arousal", "Valence"),
               model_type = fct_recode(model_type, "Looming motion" = "flynet", "Static visual quality" = "emonet"),
               partial.r.abs = sqrt(partial.r2),
               across(c(partial.r.abs, pval), \(x) signif(x, digits = 3))) %>% 
        arrange(desc(model_type), outcome) %>% 
        select(`Model type` = model_type, `Outcome` = outcome, `Partial r` = partial.r.abs, `p-value` = pval) %>% 
        write_csv(file = out_path)
      
      out_path
    },
    format = "file"
  )
)

## plotzzz ----

targets_plot_helpers <- list(
  tar_target(
    name = order_flynet_auc.by.category,
    command = preds_flynet_ckvids %>% 
      pivot_longer(cols = starts_with(".pred"), 
                   names_to = "this_emotion", 
                   values_to = "classifier_prob", 
                   names_prefix = ".pred_") %>% 
      nest(probs = -this_emotion) %>% 
      mutate(probs = map2(probs, this_emotion, 
                          \(x1, x2) mutate(x1, 
                                           across(c(emotion_obs, emotion_pred), 
                                                  \(y) fct_collapse(y, this_emotion = x2, other_level = "other")
                                           )
                          )
      )
      ) %>% 
      unnest(probs) %>% 
      group_by(this_emotion) %>% 
      roc_auc(truth = emotion_obs, classifier_prob) %>% 
      arrange(.estimate) %>% 
      pull(this_emotion)
  )
)

targets_plots <- list(
  tar_target(
    name = plot_model.acc_ckvids,
    command = perm.pvals_model.acc_ckvids %>% 
      filter(.metric == "accuracy") %>% 
      mutate(pval_text = if_else(model_type == "emonet", "p < .001", as.character(pval_text))) %>% 
      ggplot(aes(x = model_type, y = accuracy, fill = model_type)) + 
      geom_col() +
      geom_text(aes(label = pval_text), nudge_y = .01) +
      geom_hline(yintercept = .05, linetype = "dotted") +
      guides(fill = "none",
             x = guide_axis(angle = 30)) +
      labs(x = "Model type", y = "20-way emotion classification accuracy")
  ),
  tar_target(
    name = plot_model.acc.by.category_ckvids,
    command = auc.by.category_bothnets_ckvids %>% 
        mutate(.estimate = if_else(model_type == "emonet", -.estimate, .estimate)) %>% 
        ggplot(aes(x = .estimate, y = factor(this_emotion, levels = order_flynet_auc.by.category), fill = model_type)) + 
        geom_col(position = "identity") +
        geom_vline(xintercept = c(-.5, .5), linetype = "dotted") +
        # Bc the funnel-style plot must be hacked by setting one condition to negative values to flip the bars
        scale_x_continuous(labels = \(x) abs(x)) +
        # For thy fearful symmetry
        expand_limits(x = c(-1, 1)) +
        labs(x = "area under ROC curve", y = "Emotion category", fill = "Which model?")
  ),
  tar_target(
    name = plot_structure.coefs_flynet_ckvids,
    command = preds_flynet_ckvids %>% 
      left_join(rsplit_flynet_ckvids %>% 
                  testing() %>% 
                  select(-split, -starts_with("intercept")), 
                by = c("video", "emotion_obs" = "emotion", "censored")) %>% 
      pivot_longer(cols = starts_with(".pred"), 
                   names_to = "pred_class", 
                   values_to = "prob", 
                   names_prefix = ".pred_") %>% 
      pivot_longer(starts_with("slope"), 
                   names_to = "unit_num", 
                   values_to = "slope", 
                   names_prefix = "slope_", 
                   names_transform = list(unit_num = as.integer)) %>% 
      group_by(pred_class, unit_num) %>% 
      summarize(correlation = cor(prob, slope)) %>% 
      mutate(unit_x = unit_num %% 16, unit_y = unit_num %/% 16) %>% 
      ggplot(aes(x = unit_x, y = -unit_y, fill = correlation)) + 
      geom_raster() + 
      facet_wrap(~ factor(pred_class, levels = rev(order_flynet_auc.by.category))) + 
      scale_fill_viridis_c(option = "magma") +
      labs(x = NULL,
           y = NULL,
           fill = "Pearson's r")
  ),
  tar_target(
    name = plot_diff.valence.by.flynet_ckvids,
    command = {
      half_confusions <- confusions_ckvids %>% 
        halve_confusions()
      
      half_confusions %>% 
        mutate(diff_valence_resid = lm(diff_valence ~ dist_emonet, data = half_confusions) %>% 
                 pluck("residuals")) %>% 
        ggplot(aes(x = dist_flynet, y = diff_valence_resid)) + 
        geom_point() + 
        geom_smooth(method = "lm", color = "#41b6e6") +
        labs(x = "Between-category looming\nrepresentational distance",
             y = "Between-category valence\ndistance (residualized)")
    }
  ),
  tar_target(
    name = plot_diff.arousal.by.flynet_ckvids,
    command = {
      half_confusions <- confusions_ckvids %>% 
        halve_confusions()
      
      half_confusions %>% 
        mutate(diff_arousal_resid = lm(diff_arousal ~ dist_emonet, data = half_confusions) %>% 
                 pluck("residuals")) %>% 
        ggplot(aes(x = dist_flynet, y = diff_arousal_resid)) + 
        geom_point() + 
        geom_smooth(method = "lm", color = "#41b6e6") +
        labs(x = "Between-category looming\nrepresentational distance",
             y = "Between-category arousal\ndistance (residualized)")
    }
  ),
  tar_target(
    name = plot.confusion.flynet_ckvids,
    command = confusions_ckvids %>% 
      plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = dist_flynet)
  ),
  tar_target(
    name = plot.confusion.emonet_ckvids,
    command = confusions_ckvids %>% 
      plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = dist_emonet)
  ),
  tar_target(
    name = plot.confusion.valence_ckvids,
    command = confusions_ckvids %>% 
      plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = diff_valence)
  ),
  tar_target(
    name = plot.confusion.arousal_ckvids,
    command = confusions_ckvids %>% 
      plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = diff_arousal)
  )
)

targets_figs <- list(
  tar_target(
    name = fig_subjective_0564_category.probs,
    command = {
      out_filename <- "subjective_0564_category.probs.svg"
        
      (preds_flynet_ckvids %>% 
                 filter(video == "0564.mp4") %>% 
                 select(video, starts_with(".pred")) %>% 
                 pivot_longer(cols = starts_with(".pred"), 
                              names_to = "emotion_pred", 
                              values_to = "class_prob", 
                              names_prefix = ".pred_") %>% 
                 # since it's a schematic plot
                 filter(class_prob >= .001) %>% 
                 ggplot(aes(x = class_prob, y = fct_rev(emotion_pred))) + 
                 geom_col(fill = "#41b6e6") + 
                 labs(x = "Model-estimated category probability", y = NULL) +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank())) %>% 
      ggsave(filename = out_filename,
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1500,
             height = 600,
             units = "px")
      
      here::here("ignore", "figs", out_filename)
      },
    format = "file"
  ),
  tar_target(
    name = fig_model.acc_ckvids,
    command = {
      out_filename <- "subjective_model.acc_overall.svg"
      (plot_model.acc_ckvids +
                 scale_x_discrete(labels = c("Static object features", "Looming motion")) + 
                 scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") + 
                 guides(fill = "none") +
                 labs(x = NULL, y = "20-way emotion classification accuracy") + 
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank())) %>% 
      ggsave(filename = out_filename,
             plot = .,
             path = here::here("ignore", "figs"),
             width = 800,
             height = 1200,
             units = "px")
      
      here::here("ignore", "figs", out_filename)
      },
    format = "file"
  ),
  tar_target(
    name = fig_model.acc.by.category_ckvids,
    command = {
      out_filename <- "subjective_model.acc_category.svg"
      (plot_model.acc.by.category_ckvids +
       scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") + 
       annotate(geom = "text", x = -1, y = 11, label = "Static object features", angle = 90, size = 3) +
       annotate(geom = "text", x = 1, y = 11, label = "Looming motion", angle = 270, size = 3) +
       guides(fill = "none") +
       labs(y = NULL) +
       theme_bw(base_size = 12) +
       theme(plot.background = element_blank())) %>% 
      ggsave(filename = out_filename,
           plot = .,
           path = here::here("ignore", "figs"),
           width = 1600,
           height = 1000,
           units = "px")
      
      here::here("ignore", "figs", out_filename)
      },
    format = "file"
  ),
  tar_target(
    name = fig_model.acc.both_ckvids,
    command = {
      out_path <- here::here("ignore", "figs", "subjective_model.acc_both.png")
      plot_grid(
        plot_model.acc_ckvids +
          scale_x_discrete(labels = c("Static object features", "Looming motion")) + 
          scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") + 
          guides(fill = "none") +
          labs(x = NULL, y = "20-way emotion classification accuracy") + 
          theme_bw(base_size = 12) +
          theme(plot.background = element_blank()),
        plot_model.acc.by.category_ckvids +
          scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") + 
          annotate(geom = "text", x = -1, y = 11, label = "Static object features", angle = 90, size = 4) +
          annotate(geom = "text", x = 1, y = 11, label = "Looming motion", angle = 270, size = 4) +
          guides(fill = "none") +
          labs(y = NULL) +
          theme_bw(base_size = 12) +
          theme(plot.background = element_blank()),
        align = "h",
        axis = "tb",
        labels = "AUTO",
        rel_widths = c(1, 2.5)
      ) %>% 
        save_plot(filename = out_path,
                  plot = .,
                  base_height = 5,
                  base_asp = 1.618)
      
      out_path
    }
  ),
  tar_target(
    name = fig_structure.coefs_flynet_ckvids,
    command = (plot_structure.coefs_flynet_ckvids +
                 theme_bw(base_size = 12) +
                 theme(axis.text = element_blank(),
                       aspect.ratio = 1,
                       plot.background = element_blank(),
                       legend.background = element_blank())) %>% 
      ggsave(filename = "subjective_structure.coefs_flynet.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 3000,
             height = 2400,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_diff.valence.by.flynet_ckvids,
    command = (plot_diff.valence.by.flynet_ckvids +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank())) %>% 
      ggsave(filename = "subjective_diff.valence.by.flynet.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1200,
             height = 1200,
             units = "px"),
    format = "file"
  ),
  tar_target(
    name = fig_diff.arousal.by.flynet_ckvids,
    command = (plot_diff.arousal.by.flynet_ckvids +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank())) %>% 
      ggsave(filename = "subjective_diff.arousal.by.flynet.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1200,
             height = 1200,
             units = "px"),
    format = "file"
  ),
  tar_target(
    fig_confusion.emonet_ckvids,
    command = (plot.confusion.emonet_ckvids +
                 labs(x = NULL, y = NULL, fill = "Distance\n(1 - r)") +
                 scale_fill_viridis_c(option = "magma") +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank(),
                       legend.background = element_blank(),
                       aspect.ratio = 1)) %>% 
      ggsave(filename = "subjective_confusion.emonet.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 2000,
             height = 1600,
             units = "px"),
    format = "file"
  ),
  tar_target(
    fig_confusion.flynet_ckvids,
    command = (plot.confusion.flynet_ckvids +
                 labs(x = NULL, y = NULL, fill = "Distance\n(1 - r)") +
                 scale_fill_viridis_c(option = "magma") +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank(),
                       legend.background = element_blank(),
                       aspect.ratio = 1)) %>% 
      ggsave(filename = "subjective_confusion.flynet.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 2000,
             height = 1600,
             units = "px"),
    format = "file"
  ),
  tar_target(
    fig_confusion.valence_ckvids,
    command = (plot.confusion.valence_ckvids +
                 labs(x = NULL, y = NULL, fill = "Distance\n(9-pt scale units)") +
                 scale_fill_viridis_c(option = "magma") +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank(),
                       legend.background = element_blank(),
                       aspect.ratio = 1)) %>% 
      ggsave(filename = "subjective_confusion.valence.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 2000,
             height = 1600,
             units = "px"),
    format = "file"
  ),
  tar_target(
    fig_confusion.arousal_ckvids,
    command = (plot.confusion.arousal_ckvids +
                 labs(x = NULL, y = NULL, fill = "Distance\n(9-pt scale units)") +
                 scale_fill_viridis_c(option = "magma") +
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank(),
                       legend.background = element_blank(),
                       aspect.ratio = 1)) %>% 
      ggsave(filename = "subjective_confusion.arousal.png",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 2000,
             height = 1600,
             units = "px"),
    format = "file"
  )
)

## the list of all the target metanames ----

list(targets_ratings,
     targets_stimuli,
     targets_weights,
     targets_python,
     targets_preds_emonet,
     targets_activations_flynet,
     targets_preds_misc,
     targets_model_confusions_bothnets_ckvids,
     targets_stats_misc,
     targets_perms,
     target_perms.partial.r2_bothnets_ckvids,
     targets_perm_misc,
     targets_perm_results,
     target_confusions_ckvids,
     target_mds.coords_ckvids,
     targets_tables,
     targets_plot_helpers,
     targets_plots,
     targets_figs
)
