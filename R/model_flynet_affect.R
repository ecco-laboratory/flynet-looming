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
