## setup ----

require(tidymodels)
# Apparently must actively be loaded, not merely installed
require(discrim)
require(tidyverse)
require(magrittr)

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
get_discrim_preds <- function (in_data, in_workflow) {
  
  train_data <- in_data %>% 
    filter(split == "train")
  
  pred <- in_workflow %>% 
    fit(data = train_data) %>% 
    # "predict" on the FULL data
    predict(new_data = in_data) %>% 
    # bind onto the original data to recover the observation identifying columns etc
    bind_cols(in_data, .) %>% 
    rename(emotion_obs = emotion, emotion_pred = .pred_class) %>% 
    mutate(emotion_obs = factor(emotion_obs)) %>% 
    # but only keep the held-out videos from the y
    filter(split == "test") %>% 
    # Fuck the Xs! Just gimme the Ys
    select(video, censored, starts_with("emotion"))
  
  return (pred)
}

# this is basically only useful for model PERFORMANCE
# which in our case is only one of the metrics of interest
get_beh_xval_metrics <- function (in_data, in_workflow, n_folds = NULL, permute_params = NULL) {
  splits <- vfold_cv(in_data, v = n_folds, strata = emotion)
  
  # TODO: Actually use tidymodels rsample permutation implementation
  # use permutations() to generate df of permutation samples
  # map() each permutation to get a collected xval metrics within it
  # that summary of metrics will be the permutation distribution
  # Super sexy profit
  out <- in_workflow %>% 
    fit_resamples(splits) %>% 
    collect_metrics()
  
  return (out)
}

