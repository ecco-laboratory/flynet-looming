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
                      # nodelist = "node1",
                      exclude = "gpu2,node3",
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
  )
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

target_coefs.confusions_bothnets_ckvids <- tar_target(
  name = coefs.confusions_bothnets_ckvids,
  command = get_confusion_regression_coefs(confusions_ckvids,
                                           lm_formulas = list(valence = diff_valence ~ dist_flynet + dist_emonet,
                                                              arousal = diff_arousal ~ dist_flynet + dist_emonet,
                                                              fear = diff_fear ~ dist_flynet + dist_emonet,
                                                              fear_no_arousal = diff_fear ~ dist_flynet + dist_emonet + diff_arousal))
  
)

target_partial.r2.confusions_bothnets_ckvids <- tar_target(
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
            "fear_no_arousal", "dist_emonet", calc_partial_r2(half_confusions, "diff_fear", "dist_emonet", c("dist_flynet", "diff_arousal"))
    )
  }
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
# (1 min per 100 iterations, ish) thus we actually can really gain from paralleling agian
# feed n_reps_per_batch directly into permutations()
# instead of the reps argument of tar_rep
# because we only watn to call perm_beh_metrics once per iteration
# to minimize the number of times the constant ratings CSV gets read in

targets_perms <- list(
  tar_rep(
    name = perms_bothnets_ckvids,
    command = resample_beh_metrics(in_preds_flynet = preds_flynet_ckvids,
                                   in_preds_emonet = preds.videowise_emonet_ckvids,
                                   truth_col = emotion_obs,
                                   estimate_col = emotion_pred,
                                   pred_prefix = ".pred", 
                                   path_ratings = ratings_ck2017, 
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
                                   path_ratings = ratings_ck2017, 
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
    
    bind_rows(valence = valence,
              arousal = arousal,
              fear = fear,
              fear_no_arousal = fear_no_arousal,
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
    command = (preds_flynet_ckvids %>% 
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
      ggsave(filename = "subjective_0564_category.probs.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 1500,
             height = 600,
             units = "px")
  ),
  tar_target(
    name = fig_model.acc_ckvids,
    command = (plot_model.acc_ckvids +
                 scale_x_discrete(labels = c("Static object features", "Looming motion")) + 
                 scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") + 
                 guides(fill = "none") +
                 labs(x = NULL, y = "20-way emotion classification accuracy") + 
                 theme_bw(base_size = 12) +
                 theme(plot.background = element_blank())) %>% 
      ggsave(filename = "subjective_model.acc_overall.svg",
             plot = .,
             path = here::here("ignore", "figs"),
             width = 800,
             height = 1200,
             units = "px")
  ),
  tar_target(
    name = fig_model.acc.by.category_ckvids,
    (plot_model.acc.by.category_ckvids +
       scale_fill_viridis_d(begin = 0.3, end = 0.7, option = "magma") + 
       annotate(geom = "text", x = -1, y = 11, label = "Static object features", angle = 90, size = 3) +
       annotate(geom = "text", x = 1, y = 11, label = "Looming motion", angle = 270, size = 3) +
       guides(fill = "none") +
       labs(y = NULL) +
       theme_bw(base_size = 12) +
       theme(plot.background = element_blank())) %>% 
      ggsave(filename = "subjective_model.acc_category.svg",
           plot = .,
           path = here::here("ignore", "figs"),
           width = 1600,
           height = 1000,
           units = "px")
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
     target_activations_emonet.fc7_ckvids,
     target_rsplit_emonet.fc7_ckvids,
     target_preds_emonet.fc7_ckvids,
     target_weights_flynet,
     target_activations_flynet_ckvids,
     target_rsplit_flynet_ckvids,
     targets_preds_misc,
     target_coefs.confusions_bothnets_ckvids,
     target_partial.r2.confusions_bothnets_ckvids,
     targets_stats_misc,
     targets_perms,
     target_perms.partial.r2_bothnets_ckvids,
     targets_perm_misc,
     targets_perm_results,
     target_confusions_ckvids,
     target_mds.coords_ckvids,
     targets_plot_helpers,
     targets_plots,
     targets_figs
)
