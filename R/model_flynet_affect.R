## setup ----

# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

mode_char <- function (x) {
  stopifnot(is.character(x))
  x_levels = sort(unique(x))
  x_factor = factor(x)
  
  return (x_levels[which(tabulate(x_factor) == max(tabulate(x_factor)))])
}

## read in raw activations and estimate video-wise ints and slopes ----

get_flynet_activation_ck2017 <- function (file, metadata) {
  
  out <- file %>%
    read_csv() %>% 
    # inner_join keeps only videos that appear in the kragel 2019 subset of the metadata
    inner_join(metadata, by = "video") %>% 
    pivot_longer(cols = -c(video, split, frame, emotion, censored),
                 names_to = "rf",
                 values_to = "activation") %>%
    nest(activations = c(frame, activation)) %>% 
    mutate(coefs = map(activations,
                       ~lm(activation ~ scale(frame, scale = FALSE), data = .) %>% 
                         pluck("coefficients"),
                       .progress = list(name = "RF activation slopes"))) %>% 
    select(-activations) %>% 
    unnest_wider(coefs) %>% 
    rename(intercept = "(Intercept)", slope = "scale(frame, scale = FALSE)") %>% 
    pivot_wider(id_cols = c(video, emotion, split, censored),
                names_from = rf,
                values_from = c(intercept, slope))
  
  return (out)
  
}

## fit some fuckin models ----

# Separating this from the workflow function
# Because the workflows pkg does not allow passing the strings_as_factors arg in
# such that a recipe trained within fit.workflow will coerce the video ID column to factor
# and then using bake() to drop the predictor columns before binding to new predicted data
# fails to find the new factor levels of the testing videos bc duh
get_discrim_recipe <- function (in_data) {
  # remember, data only needs something with the right colnames and types
  out <- recipe(emotion ~ ., data = head(in_data)) %>% 
    update_role(video, censored, new_role = "ID") %>% 
    step_rm(split)
  
  return (out)
}

# UNTRAINED workflow!
get_discrim_workflow <- function (in_recipe, discrim_engine = discrim_linear()) {
  
  out <- workflow() %>% 
    # By default, LDA with engine = "MASS" with discrim_linear()
    # Have to use pls(method = "classification") to get PLS-DA
    add_model(discrim_engine) %>% 
    add_recipe(in_recipe)
  
  return (out)
}

# what do we want? predictions!
# when do we want them? now!
# But don't use this for model performance bc we have tidymodels xval for that (below)
get_discrim_preds_from_trained_model <- function (in_workflow_fit, in_recipe, test_data) {
  
  pred_classes <- in_workflow_fit %>% 
    # "predict" on the HELD OUT data
    predict(new_data = test_data, type = "class")
  
  # Egh we need the class probs to calculate AUC
  pred_probs <- in_workflow_fit %>% 
    predict(new_data = test_data, type = "prob")
  
  out <- in_recipe %>%
    # use recipes prep and bake to drop the predictor cols
    # this should be evergreen to changes in the names of the predictor cols
    prep(training = NULL, strings_as_factors = FALSE) %>% 
    bake(test_data, has_role("ID"), all_outcomes()) %>% 
    # bind back onto the original df to recover the observation identifying columns etc
    # apparently this is how they do it in the tidymodels docs too, so there isn't a better way
    # so... dumb
    bind_cols(pred_classes, pred_probs) %>% 
    rename(emotion_obs = emotion, emotion_pred = .pred_class) %>% 
    mutate(emotion_obs = factor(emotion_obs))
  
  return (out)
}

get_confusion_regression_coefs <- function (confusions, lm_formulas) {
  half_confusions <- confusions %>% 
    halve_confusions()
  
  out <- tibble(outcome = names(lm_formulas),
                lm_formula = lm_formulas) %>% 
    mutate(coefs = map(lm_formula,
                       \(x) lm(x, data = half_confusions) %>% 
                         broom::tidy()
                       )
           ) %>% 
    select(-lm_formula) %>% 
    unnest(coefs)
  
  return (out)
}

## making confusion matrices from model preds ----

make_full_confusions <- function (preds_flynet, preds_emonet, dists_ratings) {

  confusions_flynet <- calc_distances_modelprobs(preds_flynet)
  confusions_emonet <- calc_distances_modelprobs(preds_emonet)
  
  # TODO: For permutation testing these statistics, 
  # each perm iteration should have a confusion matrix calculated from a set 
  # with the testing observed labels permuted. 
  # The same type of permutation being done to estimate class accuracy
  
  out <- confusions_flynet %>% 
    full_join(confusions_emonet,
              by = c("emotion_obs", "emotion_pred"),
              suffix = c("_flynet", "_emonet")) %>% 
    mutate(fear_only = case_when(
             emotion_obs == "Fear" & emotion_pred == "Fear" ~ 0L,
             emotion_obs != "Fear" & emotion_pred != "Fear" ~ 0L,
             TRUE ~ 1L
           ),
           active_avoidance = case_when(
             emotion_obs %in% c("Fear", "Horror", "Disgust") & emotion_pred %in% c("Fear", "Horror", "Disgust") ~ 0L,
             !(emotion_obs %in% c("Fear", "Horror", "Disgust")) & !(emotion_pred %in% c("Fear", "Horror", "Disgust")) ~ 0L,
             TRUE ~ 1L
           )
    ) %>% 
    # left instead of full join to ignore emotion categories not in the EmoNet set
    left_join(dists_ratings,
              by = c("emotion_obs", "emotion_pred"))
  
  return (out)
}

calc_confusions <- function (preds) {
  out <- preds %>% 
    count(emotion_obs, emotion_pred) %>% 
    # IMPORTANT!!!
    # This works for FlyNet AND EmoNet,
    # even though EmoNet has no instances of empathic pain predicted
    # ONLY IF the emotion columns are FACTOR
    # with ALL 20 levels (even if one of the levels is never observed)
    # AND THEY ARE NOW!!! SO DON'T SWITCH IT BACK
    complete(emotion_obs, emotion_pred, fill = list(n = 0L)) %>% 
    group_by(emotion_obs) %>% 
    mutate(prob = n / sum(n),
           dist = 1 - prob) %>% 
    ungroup() %>% 
    select(-n)
  
  return (out)
}

halve_confusions <- function (confusions) {
  out <- confusions %>% 
    # get the bottom half, WITH diagonals, of the matrix
    mutate(across(starts_with("emotion"), \(x) as.integer(factor(x)))) %>% 
    filter(emotion_obs <= emotion_pred)
  
  return (out)
}

calc_distances_modelprobs <- function (preds) {
  out <- preds %>% 
    pivot_longer(cols = starts_with(".pred"), 
                 names_to = "emotion", 
                 values_to = "prob", 
                 names_prefix = ".pred_") %>% 
    select(-emotion, -emotion_pred) %>% 
    chop(prob) %>% 
    rename(video_obs = video, prob_obs = prob) %>% 
    mutate(video_pred = video_obs, emotion_pred = emotion_obs, prob_pred = prob_obs) %>% 
    complete(nesting(video_obs, emotion_obs, prob_obs), 
             nesting(video_pred, emotion_pred, prob_pred)) %>% 
    # note that this WILL produce a symmetric correlation matrix
    mutate(correlation = map2_dbl(prob_obs, prob_pred, \(x, y) cor(x, y, method = "pearson"))) %>% 
    select(-starts_with("prob")) %>% 
    filter(video_obs != video_pred) %>% 
    group_by(emotion_obs, emotion_pred) %>% 
    summarize(correlation = abs(mean(correlation)), .groups = "drop") %>% 
    mutate(dist = 1 - correlation) %>% 
    select(-correlation)
  
  return (out)
}

calc_distances_ratings <- function (path_ratings, path_ids_filter = NULL) {
  # The target is a path to file so focus on making that the function arg
  # versus having them each take a df as input
  out <- read_csv(path_ratings) %>% 
    select(video = Filename, arousal = arousal...37, valence, fear = Fear)
  
  if (!is.null(path_ids_filter)) {
    out %<>%
      # for example, keep only the TRAINING videos
      # so this has the effect of "fitting" a "model" on the training videos
      inner_join(read_csv(path_ids_filter), 
                 by = "video")
  }
  
  out %<>% 
    # first calc the mean rating for each emotion category
    group_by(emotion) %>% 
    summarize(across(c(arousal, valence, fear), mean)) %>% 
    # then copy the cols so we can get category pairwise distances
    mutate(across(everything(), \(x) x, .names = "{.col}_pred")) %>% 
    rename_with(\(x) paste0(x, "_obs"), -ends_with("pred")) %>% 
    complete(nesting(emotion_obs, valence_obs, arousal_obs, fear_obs), 
             nesting(emotion_pred, valence_pred, arousal_pred, fear_pred)) %>% 
    mutate(diff_arousal = abs(arousal_obs - arousal_pred),
           diff_valence = abs(valence_obs - valence_pred),
           diff_fear = abs(fear_obs - fear_pred)) %>% 
    # keep only the diff cols
    select(starts_with("emotion"), starts_with("diff"))
  
  return (out)
}

# takes the long confusions from the above function and turns it into a dist object
convert_long_to_dist <- function (distances, row_col, col_col, y_col, flip_dist = TRUE) {
  row_col <- enquo(row_col)
  col_col <- enquo(col_col)
  y_col <- enquo(y_col)
  
  # If the input was not previously symmetrical,
  # this will forcibly average the corresponding cells and turn it into a half/triangle matrix
  # If the input was ALREADY symmetrical,
  # this should only have the effect of turning it into a half/triangle matrix
  # Either way, that half matrix is what as.dist() can coerce to a distance object
  
  pre_dist <- distances %>% 
    # First, break the row-column association in preparation for averaging across triangles of the matrix
    mutate(across(c(!!row_col, !!col_col), as.character), 
           cols = map2(!!row_col, !!col_col,
                       \(x, y) set_names(sort(c(x, y)), nm = c("id1", "id2")))) %>%
    unnest_wider(cols) %>% 
    # This stuff actually does the averaging across
    group_by(id1, id2) %>% 
    # take advantage of this moment to clear up later code by not having to unwrap a col name as variable
    summarize(y = mean(!!y_col))
  
  if (flip_dist) {
    pre_dist %<>% 
      mutate(y = 1 - y)
  }
  
  out <- pre_dist %>% 
    pivot_wider(id_cols = id2, names_from = id1, values_from = y) %>%
    column_to_rownames("id2") %>% 
    # This assumes the diagonal dissimilarity is 0, which is most definitely not true...
    as.dist(diag = TRUE)
  
  return (out)
}

# This is only for plotting symmetrized distances at the moment
# It keeps both triangles of the matrix otherwise the triangleyness depends on the ordering of levels
# which very much is not handled by this
symmetrize_distances <- function (distances, row_col, col_col, y_cols) {
  row_col <- enquo(row_col)
  col_col <- enquo(col_col)
  y_cols <- enquo(y_cols)
  
  out <- distances %>% 
    # First, break the row-column association in preparation for averaging across triangles of the matrix
    mutate(across(c(!!row_col, !!col_col), as.character), 
           cols = map2(!!row_col, !!col_col,
                       \(x, y) set_names(sort(c(x, y)), nm = c("id1", "id2")))) %>%
    unnest_wider(cols) %>% 
    # This stuff actually does the averaging across
    group_by(id1, id2) %>% 
    mutate(across(!!y_cols, mean)) %>% 
    ungroup() %>% 
    select(-id1, -id2)
  
  return (out)
}

## permutation testing but tidy this time ----

# this is basically only useful for model PERFORMANCE
# which in our case is only one of the metrics of interest
# Phil says we don't need to cross-validate model performance because we have enough videos
# that the estimate from a single train-test split should be fairly stable/low SE
# Note that times sets the number of permutation iterations to be run in a single batch
# and there is no paralleling inside of this function
# but since it's not re-fitting any models, it's pretty fast and doesn't NEED to be paralleled
resample_beh_metrics <- function (in_preds_flynet, 
                                  in_preds_emonet, 
                                  truth_col, 
                                  estimate_col, 
                                  pred_prefix, 
                                  path_ratings, 
                                  path_ids_train,
                                  resample_type = "permute",
                                  times) {
  truth_col <- enquo(truth_col)
  estimate_col <- enquo(estimate_col)
  
  # hold onto it so that we can compare the permuted pred distances
  # with the valence and arousal distances
  rating_distances <- calc_distances_ratings(path_ratings, path_ids_train)
  
  # Per Kragel 2019 paper, this takes finished preds,
  # which effectively keeps the training model the same, without refitting it
  out <- in_preds_flynet %>% 
    select(-censored) %>% 
    left_join(in_preds_emonet, by = c("video", "emotion_obs"), suffix = c(".flynet", ".emonet"))
  
  if (resample_type == "permute") {
    out %<>%
      # permute the ground truth outcomes before calculating classification "accuracy"
      permutations(permute = !!truth_col, times = times)
  } else if (resample_type == "bootstrap") {
    out %<>%
      # bootstrap resample before calculating classification accuracy
      bootstraps(permute = !!truth_col, times = times)
  }
  out %<>% 
    # needs to be named using tidymodels tune convention to use tune metrics collecting later
    mutate(.metrics_flynet = map(splits, 
                        \(x) x %>% 
                          analysis() %>% 
                          select(video, !!truth_col, ends_with(".flynet")) %>% 
                          rename_with(\(x) str_sub(x, end = -8L), ends_with(".flynet")) %>% 
                          # this should get us accuracy and multi-class AUC
                          # from the default tidymodels classification perf metrics
                          # mind that AUC makes this take measurably longer
                          # 1 min for 100 iterations vs just a couple seconds before
                          metrics(truth = !!truth_col, 
                                  estimate = !!estimate_col, 
                                  starts_with(pred_prefix)),
                          .progress = "resampling FlyNet performance"),
           .metrics_emonet = map(splits, 
                                 \(x) x %>% 
                                   analysis() %>% 
                                   select(video, !!truth_col, ends_with(".emonet")) %>% 
                                   rename_with(\(x) str_sub(x, end = -8L), ends_with(".emonet")) %>% 
                                   metrics(truth = !!truth_col, 
                                           estimate = !!estimate_col, 
                                           starts_with(pred_prefix)),
                                 .progress = "resampling EmoNet performance"))
  
  return (out)
}
