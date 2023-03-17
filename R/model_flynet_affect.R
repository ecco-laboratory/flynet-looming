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

get_discrim_workflow <- function (in_data, discrim_engine = discrim_linear()) {
  # remember, data only needs something with the right colnames and types
  discrim_recipe <- recipe(emotion ~ ., data = head(in_data)) %>% 
    update_role(video, censored, new_role = "ID") %>% 
    step_rm(split)
  
  discrim_workflow <- workflow() %>% 
    # By default, LDA with engine = "MASS" with discrim_linear()
    # Have to use pls(method = "classification") to get PLS-DA
    add_model(discrim_engine) %>% 
    add_recipe(discrim_recipe)
  
  return (discrim_workflow)
}

# what do we want? predictions!
# when do we want them? now!
# But don't use this for model performance bc we have tidymodels xval for that (below)
get_discrim_preds_from_trained_model <- function (in_workflow_fit, test_data) {
  
  out <- in_workflow_fit %>% 
    # "predict" on the HELD OUT data
    predict(new_data = test_data) %>% 
    # bind back onto the original df to recover the observation identifying columns etc
    # apparently this is how they do it in the tidymodels docs too, so there isn't a better way
    # so... dumb
    bind_cols(test_data, .) %>% 
    rename(emotion_obs = emotion, emotion_pred = .pred_class) %>% 
    mutate(emotion_obs = factor(emotion_obs)) %>% 
    # Fuck the Xs! Just gimme the Ys
    select(video, censored, starts_with("emotion"))
  
  return (out)
}

## permutation testing but tidy this time ----

# this is basically only useful for model PERFORMANCE
# which in our case is only one of the metrics of interest
# Phil says we don't need to cross-validate model performance because we have enough videos
# that the estimate from a single train-test split should be fairly stable/low SE
# Note that times sets the number of permutation iterations to be run in a single batch
# and there is no paralleling inside of this function
perm_beh_metrics <- function (in_split, in_workflow, times) {
  
  out <- in_split %>% 
    # Only permute emotion classes for the training data
    # ne touche pas les testing data. that's for prediction only!
    training() %>% 
    permutations(permute = emotion, times = times) %>% 
    # needs to be named using tidymodels tune convention to use tune metrics collecting later
    mutate(.metrics = map(splits, 
                        \(x) x %>% 
                          analysis() %>% 
                          fit(in_workflow, data = .) %>% 
                          # testing on the same held-out, NOT PERMUTED, testing data every time
                          get_discrim_preds_from_trained_model(test_data = testing(in_split)) %>% 
                          accuracy(truth = emotion_obs, estimate = emotion_pred),
                          .progress = "permutation! no breathing!"))
  
  return (out)
}
