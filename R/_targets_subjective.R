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
  packages = c("mixOmics",
               "tidymodels",
               "plsmod",
               "discrim",
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
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.

# plan(multicore)
# eventually... when I figure out why slurm is activating R 3.6.3
n_slurm_cpus <- 1L
plan(batchtools_slurm,
     template = "future.tmpl",
     resources = list(ntasks = 1L,
                      ncpus = n_slurm_cpus,
                      nodelist = "node1",
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 1800L,
                      memory = 500L,
                      partition = "short"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 50
n_reps_per_batch <- 200

# Run the R scripts in the R/ folder with your custom functions:
tar_source(c("R/model_flynet_affect.R",
             "R/plot_helpers.R"
))

# source("other_functions.R") # Source other scripts as needed. # nolint

## metadata files from other people's stuff ----

target_ratings_ck2017 <- tar_target(
  name = ratings_ck2017,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/video_ratings.csv",
  format = "file"
)

target_censored_ck2017 <- tar_target(
  name = censored_ck2017,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/censored_video_ids.csv",
  format = "file"
)

target_ids.train_kragel2019 <- tar_target(
  name = ids.train_kragel2019,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/train_video_ids.csv",
  format = "file"
)

target_ids.test_kragel2019 <- tar_target(
  name = ids.test_kragel2019,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/test_video_ids.csv",
  format = "file"
)

target_classes_ck2017 <- tar_target(
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
)

target_weights_zhou2022 <- tar_map(
  values = tibble(filename = list.files(here::here("ignore",
                                                   "models",
                                                   "zhou2022"))),
  tar_target(name = weights_zhou2022,
             command = here::here("ignore",
                                  "models",
                                  "zhou2022",
                                  filename),
             format = "file")
)

## python scripts ----

target_py_flynet_utils <- tar_target(
  name = py_flynet_utils,
  command = here::here("python",
                       "myutils",
                       "flynet_utils.py"),
  format = "file"
)

target_py_convert_flynet_weights <- tar_target(
  name = py_convert_flynet_weights,
  command = here::here("python",
                       "myutils",
                       "convert_flynet_weights.py"),
  format = "file"
)

target_py_calc_flynet_activations <- tar_target(
  name = py_calc_flynet_activations,
  command = here::here("python",
                       "myutils",
                       "calc_flynet_activations.py"),
  format = "file"
)

## emonet dependent stuff ----

# TODO: Fully implement this from the python script
target_preds.framewise_emonet_ckvids <- tar_target(
  name = preds.framewise_emonet_ckvids,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/test_framewise_preds.csv",
  format = "file"
)

target_preds.videowise_emonet_ckvids <- tar_target(
  name = preds.videowise_emonet_ckvids,
  command = {
    out <- read_csv(preds.framewise_emonet_ckvids) %>% 
      select(-guess_1) %>% 
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
             # Pull the levels from the observed labels ecause empathic pain was never guessed
             emotion_pred = factor(emotion_pred, levels = levels(emotion_obs)))
  }
)

target_activations_emonet.fc7_ckvids <- tar_target(
  name = activations_emonet.fc7_ckvids,
  command = "/home/mthieu/Repos/CowenKeltner/metadata/kragel2019_videowise_activations_fc7.csv",
  format = "file"
)

## flynet setup stuff ----

target_weights_flynet <- tar_target(
  name = weights_flynet,
  command = {
    inject(!!syms(paste0("weights_zhou2022_", 
                list.files(here::here("ignore",
                                      "models",
                                      "zhou2022")))))
    system2("python", args = c(py_convert_flynet_weights, "-u 256"))
    here::here("ignore",
               "models",
               "MegaFlyNet256.pt")
  },
  format = "file"
)

## flynet activations ----

target_activations_flynet_ckvids <- tar_target(
  name = activations_flynet_ckvids,
  command = {
    weights_flynet
    system2("python",
            args = c(py_calc_flynet_activations,
                     "-l 132",
                     "-p /home/mthieu/Repos/CowenKeltner",
                     "-v videos_10fps",
                     "-m metadata",
                     "-q activations"))
    "/home/mthieu/Repos/CowenKeltner/metadata/flynet_132x132_stride8_activations.csv"
  },
  format = "file"
)

## beh model fitting ----

target_rsplit_flynet_ckvids <- tar_target(
  name = rsplit_flynet_ckvids,
  command = {
    activations <- get_flynet_activation_ck2017(activations_flynet_ckvids, classes_ck2017)
    # Use pre-existing Kragel 2019 train-test split to make an rsample-compatible split object
    make_splits(x = filter(activations, split == "train"),
                assessment = filter(activations, split == "test"))
  }
)

target_preds_flynet_ckvids <- tar_target(
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
)

target_rsplit_emonet.fc7_ckvids <- tar_target(
  name = rsplit_emonet.fc7_ckvids,
  command = {
    
    activations <- read_csv(activations_emonet.fc7_ckvids) %>% 
      # selects 256 units to get a model with the same sparsity as FlyNet
      # It's not random because I didn't want to set a target-specific seed
      select(video, !!seq(17, 4097, length.out = 256)) %>% 
      rename_with(\(x) paste0("unit_", x),.cols = -video) %>% 
      inner_join(classes_ck2017, by = "video")
    # Use pre-existing Kragel 2019 train-test split to make an rsample-compatible split object
    make_splits(x = filter(activations, split == "train"),
                assessment = filter(activations, split == "test"))
  }
)

target_preds_emonet.fc7_ckvids <- tar_target(
  name = preds_emonet.fc7_ckvids,
  command = {
    discrim_recipe <- rsplit_emonet.fc7_ckvids %>% 
      training() %>% 
      get_discrim_recipe()
    
    discrim_recipe %>% 
      get_discrim_workflow() %>% 
      fit(data = training(rsplit_emonet.fc7_ckvids)) %>% 
      get_discrim_preds_from_trained_model(in_recipe = discrim_recipe,
                                           test_data = testing(rsplit_emonet.fc7_ckvids))
  }
)

## beh permutation testing ----

# Note that these are no longer paralleled using tar_rep
# as perm_beh_metrics() now takes finished predictions and only permutes the true outcomes
# leaving the predictions (and effectively the training model) the same every time
# speeding up the operation mightily as models don't need to be refit
target_perms_flynet_ckvids <- tar_target(
  name = perms_flynet_ckvids,
  command = perm_beh_metrics(in_preds = preds_flynet_ckvids,
                             truth_col = emotion_obs,
                             estimate_col = emotion_pred,
                             times = n_batches * n_reps_per_batch) %>% 
    select(-splits)
)

target_perms_emonet_ckvids <- tar_target(
  name = perms_emonet_ckvids,
  command = perm_beh_metrics(in_preds = preds.videowise_emonet_ckvids,
                             truth_col = emotion_obs,
                             estimate_col = emotion_pred,
                             times = n_batches * n_reps_per_batch) %>% 
    select(-splits)
)

target_perms_emonet.fc7_ckvids <- tar_target(
  name = perms_emonet.fc7_ckvids,
  command = perm_beh_metrics(in_preds = preds_emonet.fc7_ckvids,
                             truth_col = emotion_obs,
                             estimate_col = emotion_pred,
                             times = n_batches * n_reps_per_batch) %>% 
    select(-splits)
)

## long-form confusions and distances ----

target_confusions_ckvids <- tar_target(
  name = confusions_ckvids,
  command = make_full_confusions(preds_flynet_ckvids,
                                 preds.videowise_emonet_ckvids,
                                 ratings_ck2017,
                                 ids.train_kragel2019)
)

target_mds.coords_ckvids <- tar_target(
  name = mds.coords_ckvids,
  command = {
    rating_means <- read_csv(ratings_ck2017) %>% 
      select(video = Filename, arousal = arousal...37, valence) %>% 
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
                                     y_col = prob_emonet),
       flynet = convert_long_to_dist(distances = confusions_ckvids,
                                     row_col = emotion_obs,
                                     col_col = emotion_pred,
                                     y_col = prob_flynet)) %>% 
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

## plotzzz ----

target_plot_model.acc_ckvids <- tar_target(
  name = plot_model.acc_ckvids,
  command = {
    class_accuracies <- bind_rows(flynet = preds_flynet_ckvids,
                                  emonet = preds.videowise_emonet_ckvids,
                                  .id = "model_type") %>% 
      group_by(model_type) %>% 
      accuracy(truth = emotion_obs, estimate = emotion_pred)
    
    class_accuracy_pvals <- bind_rows(flynet = perms_flynet_ckvids,
                                      emonet = perms_emonet_ckvids,
                                      .id = "model_type") %>% 
      unnest(.metrics) %>% 
      rename(.estimate_perm = .estimate) %>% 
      left_join(class_accuracies, by = "model_type") %>% 
      group_by(model_type) %>% 
      # Carrying the accuracy value through so we can plot everything through this df alone
      summarize(accuracy = mean(.estimate),
                pval = (sum(.estimate_perm >= .estimate) + 1) / (n() + 1)) %>% 
      mutate(pval_text = glue::glue("p = {signif(pval, digits = 2)}"))
    
    class_accuracy_pvals %>% 
      ggplot(aes(x = model_type, y = accuracy, fill = model_type)) + 
      geom_col() +
      geom_text(aes(label = pval_text), data = class_accuracy_pvals, nudge_y = .01) +
      geom_hline(yintercept = .05, linetype = "dotted") +
      guides(fill = "none") +
      labs(x = "Model type", y = "20-way emotion classification accuracy")
  }
)

target_plot_model.acc.by.category_ckvids <- tar_target(
  name = plot_model.acc.by.category_ckvids,
  command = {
    flynet_acc_by_category_order <- preds_flynet_ckvids %>% 
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
    
    bind_rows(flynet = preds_flynet_ckvids, 
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
      unnest(probs) %>% 
      group_by(model_type, this_emotion) %>% 
      roc_auc(truth = emotion_obs, classifier_prob) %>% 
      mutate(.estimate = if_else(model_type == "emonet", -.estimate, .estimate)) %>% 
      ggplot(aes(x = .estimate, y = factor(this_emotion, levels = flynet_acc_by_category_order), fill = model_type)) + 
      geom_col(position = "identity") +
      geom_vline(xintercept = c(-.5, .5), linetype = "dotted") +
      # Bc the funnel-style plot must be hacked by setting one condition to negative values to flip the bars
      scale_x_continuous(labels = \(x) abs(x)) +
      # For thy fearful symmetry
      expand_limits(x = c(-1, 1)) +
      labs(x = "area under ROC curve", y = "Emotion category", fill = "Which model?")
  }
)

target_plot.confusion.flynet_ckvids <- tar_target(
  name = plot.confusion.flynet_ckvids,
  command = confusions_ckvids %>% 
      symmetrize_distances(row_col = emotion_obs, col_col = emotion_pred, y_col = prob_flynet) %>% 
      plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = prob_flynet)
)

target_plot.confusion.emonet_ckvids <- tar_target(
  name = plot.confusion.emonet_ckvids,
  command = confusions_ckvids %>% 
    symmetrize_distances(row_col = emotion_obs, col_col = emotion_pred, y_col = prob_emonet) %>% 
    plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = prob_emonet)
)

target_plot.confusion.valence_ckvids <- tar_target(
  name = plot.confusion.valence_ckvids,
  command = confusions_ckvids %>% 
    plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = diff_valence)
)

target_plot.confusion.arousal_ckvids <- tar_target(
  name = plot.confusion.arousal_ckvids,
  command = confusions_ckvids %>% 
    plot_confusion_matrix(row_col = emotion_obs, col_col = emotion_pred, fill_col = diff_arousal)
)

## the list of all the target metanames ----

list(target_ratings_ck2017,
     target_censored_ck2017,
     target_ids.train_kragel2019,
     target_ids.test_kragel2019,
     target_classes_ck2017,
     target_weights_zhou2022,
     target_py_flynet_utils,
     target_py_convert_flynet_weights,
     target_py_calc_flynet_activations,
     target_preds.framewise_emonet_ckvids,
     target_preds.videowise_emonet_ckvids,
     target_perms_emonet_ckvids,
     target_activations_emonet.fc7_ckvids,
     target_rsplit_emonet.fc7_ckvids,
     target_preds_emonet.fc7_ckvids,
     target_perms_emonet.fc7_ckvids,
     target_weights_flynet,
     target_activations_flynet_ckvids,
     target_rsplit_flynet_ckvids,
     target_preds_flynet_ckvids,
     target_perms_flynet_ckvids,
     target_confusions_ckvids,
     target_mds.coords_ckvids,
     target_plot_model.acc_ckvids,
     target_plot_model.acc.by.category_ckvids,
     target_plot.confusion.flynet_ckvids,
     target_plot.confusion.emonet_ckvids,
     target_plot.confusion.valence_ckvids,
     target_plot.confusion.arousal_ckvids
)
