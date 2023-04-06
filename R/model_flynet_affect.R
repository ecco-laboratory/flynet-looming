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

## making confusion matrices from model preds ----

make_full_confusions <- function (preds_flynet, preds_emonet, path_ratings, path_ids_train) {
  # As silly as this is, I think this would only get called in here,
  # so I don't see huge utility in externalizing this bit
  confusions_flynet <- preds_flynet %>% 
    count(emotion_obs, emotion_pred) %>% 
    complete(emotion_obs, emotion_pred, fill = list(n = 0L)) %>% 
    group_by(emotion_obs) %>% 
    mutate(prob = n / sum(n)) %>% 
    ungroup()
  
   # Especially bc it's not exactly the same for EmoNet
  confusions_emonet <- preds_emonet %>% 
    count(emotion_obs, emotion_pred) %>% 
    complete(emotion_obs, emotion_pred, fill = list(n = 0L)) %>% 
    # No videos were predicted as empathic pain
    # so need to do this shit to patch it back in
    pivot_wider(id_cols = emotion_obs,
                names_from = emotion_pred,
                values_from = n) %>% mutate(`Empathic Pain` = 0L) %>% 
    pivot_longer(cols = -emotion_obs,
                 names_to = "emotion_pred",
                 values_to = "n") %>% 
    group_by(emotion_obs) %>% 
    mutate(prob = n / sum(n)) %>% 
    ungroup()
  
  rating_means <- read_csv(path_ratings) %>% 
    select(video = Filename, arousal = arousal...37, valence) %>% 
    # Keep only the TRAINING videos
    # so this has the effect of "fitting" a "model" on the training videos
    inner_join(read_csv(path_ids_train), 
               by = "video") %>% 
    group_by(emotion) %>% 
    summarize(across(c(arousal, valence), mean))
  
  # TODO: For permutation testing these statistics, 
  # each perm iteration should have a confusion matrix calculated from a set 
  # with the testing observed labels permuted. 
  # The same type of permutation being done to estimate class accuracy
  
  out <- confusions_flynet %>% 
    select(-n) %>% 
    rename(prob_flynet = prob) %>% 
    full_join(confusions_emonet %>% 
                select(-n) %>% 
                rename(prob_emonet = prob),
              by = c("emotion_obs", "emotion_pred")) %>% 
    mutate(dist_flynet = 1 - prob_flynet,
           dist_emonet = 1 - prob_emonet,
           fear_only = case_when(
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
    # eh yeah paste it on twice for the observed and the predicted...
    # not glam but I can't think of a more glam option
    left_join(rating_means %>% rename_with(\(x) paste0(x, "_observed"),
                                           .cols = -emotion),
              by = c("emotion_obs" = "emotion")) %>% 
    left_join(rating_means %>% rename_with(\(x) paste0(x, "_predicted"),
                                           .cols = -emotion),
              by = c("emotion_pred" = "emotion")) %>% 
    mutate(diff_arousal = abs(arousal_observed - arousal_predicted),
           diff_valence = abs(valence_observed - valence_predicted)) %>% 
    # keep only the diff cols
    select(-ends_with("observed"), -ends_with("predicted"))
  
  return (out)
}

# This is only for plotting symmetrized distances at the moment
# It keeps both triangles of the matrix otherwise the triangleyness depends on the ordering of levels
# which very much is not handled by this
symmetrize_distances <- function (distances, row_col, col_col, y_col) {
  row_col <- enquo(row_col)
  col_col <- enquo(col_col)
  y_col <- enquo(y_col)
  
  out <- distances %>% 
    # First, break the row-column association in preparation for averaging across triangles of the matrix
    mutate(across(c(!!row_col, !!col_col), as.character), 
           cols = map2(!!row_col, !!col_col,
                       \(x, y) set_names(sort(c(x, y)), nm = c("id1", "id2")))) %>%
    unnest_wider(cols) %>% 
    # This stuff actually does the averaging across
    group_by(id1, id2) %>% 
    mutate(!!y_col := mean(!!y_col)) %>% 
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
perm_beh_metrics <- function (in_preds, truth_col, estimate_col, times) {
  truth_col <- enquo(truth_col)
  estimate_col <- enquo(estimate_col)
  # Per Kragel 2019 paper, this takes finished preds,
  # which effectively keeps the training model the same, without refitting it
  
  out <- in_preds %>% 
    # permute the ground truth outcomes before calculating classification "accuracy"
    permutations(permute = !!truth_col, times = times) %>% 
    # needs to be named using tidymodels tune convention to use tune metrics collecting later
    mutate(.metrics = map(splits, 
                        \(x) x %>% 
                          analysis() %>% 
                          accuracy(truth = !!truth_col, estimate = !!estimate_col),
                          .progress = "permutation! no breathing!"))
  
  return (out)
}
