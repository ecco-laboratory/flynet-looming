## setup ----

# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

## permutation by time ----

# Returns a df with ONLY old (true) TR, and permuted TR
# to be used for feeding into get_pls_metrics() so that the same randomization
# can be applied across different xval folds with the same iteration number
# and where the timecourse is permuted in the same way for each subject!!!
get_permuted_order <- function (in_y, n_cycles) {
  
  # 0 if the first TR is 1
  tr_start_offset <- min(in_y$tr_num) - 1L
  
  if (n_cycles > 1) {
    block_durations <- in_y %>% 
      count(subj_num, run_type, run_num, stim_type) %>% 
      pull(n) %>% 
      unique()
    
    # Do not know what to do if cycle length is not an integer number of TRs
    # Should ideally never happen
    cycle_lengths <- block_durations / n_cycles
    stopifnot(all(cycle_lengths %% 1 == 0))
  }
  
  out <- in_y %>% 
    select(run_type, run_num, stim_type, tr_num) %>% 
    distinct() %>% 
    group_by(run_type, run_num, stim_type) %>% 
    # a grouping variable for cycle num will make it easier
    # to permute within each stimulus cycle if there are multiple cycles in a stim-type block/run
    # adjust so the TR num starts counting from 0, then divide... evenly?
    mutate(cycle_length = n() / n_cycles,
           cycle_num = (tr_num - (tr_start_offset + 1)) %/% cycle_length) %>% 
    group_by(run_type, run_num, stim_type, cycle_length, cycle_num) %>% 
    # this one actually does the permuting! should shuffle only TR within cycle, basically
    slice_sample(prop = 1) %>% 
    mutate(tr_num_new = 1:n() + (cycle_length * cycle_num) + tr_start_offset) %>% 
    ungroup() %>% 
    select(-starts_with("cycle"))
  
  return (out)
}

## fits the models eek ----

# Fit model and extract predicted BOLD for a single iteration of data
get_pls_preds <- function (in_x, in_y, test_subjs, pls_num_comp = 20L, include_fit = TRUE) {
  
  in_y %<>%
    mutate(split_type = if_else(subj_num %in% test_subjs, "test", "train"))
  
  in_data <- in_x %>% 
    # Feb 8 2023 YES joining by run because we're now fitting the model across everybody's data
    # join only by run_type and not by run_num because in_x only has one instance of each run/stim type
    right_join(in_y, by = c("run_type", "tr_num")) %>% 
    select(split_type, subj_num, run_type, run_num, stim_type, tr_num, everything())
  
  train_data <- in_data %>% 
    filter(split_type == "train")
  
  pls_recipe <- recipe(head(train_data)) %>% 
    update_role(starts_with("voxel"), new_role = "outcome") %>% 
    update_role(starts_with("unit"), new_role = "predictor") %>% 
    update_role(c(subj_num, run_type, run_num), new_role = "ID") %>% 
    # because the input x for prediction might or might not actually be subj-specific
    update_role_requirements(role = "ID", bake = FALSE) %>% 
    update_role_requirements(role = NA, bake = FALSE)
  
  pls_workflow <- workflow() %>% 
    # mixOmics is the only available engine so just leave it
    add_model(parsnip::pls(mode = "regression",
                           predictor_prop = 1,
                           num_comp = pls_num_comp)) %>% 
    add_recipe(pls_recipe)
  
  # get and process model preds
  pls_fit <- pls_workflow %>% 
    fit(data = train_data)
  
  pred <- pls_fit %>% 
    # "predict" on the FULL x
    predict(new_data = in_x) %>% 
    # first bind onto the x to recover the observation identifying columns
    bind_cols(in_x, .) %>% 
    select(run_type, tr_num, starts_with(".pred")) %>% 
    # bind onto the y to compare predicted and actual
    right_join(in_y, by = c("run_type", "tr_num")) %>% 
    # but only keep the held-out subjects from the y
    filter(split_type == "test") %>% 
    select(-split_type) %>% 
    # so the obs and the pred columns both have a prefix
    rename_with(.fn = \(x) paste0("obs_", x), .cols = starts_with("voxel_")) %>% 
    pivot_longer(cols = contains(paste0("_voxel_")), 
                 names_to = c(".value", "voxel_num"), 
                 names_pattern = paste0("(.*)_voxel_(.*)"),
                 names_transform = list(voxel_num = as.integer)) %>% 
    rename(pred = .pred)
  
  if (include_fit) {
    out <- list(fit = pls_fit,
                pred = pred)
  } else {
    out <- list(pred = pred)
  }
  
  return (out)
}

# permute_params = NULL means DO NOT PERMUTE!!!
# otherwise, it should be list() with n_cycles
# permutation now lives here because we are permuting the held-out testing data
# and NOT the data going into the model
# so that we can hold the model constant and not retrain it
wrap_pred_metrics <- function (df_xval, in_groupavg, permute_order = NULL, decoding = TRUE) {
  if (!is.null(permute_order)) {
    # permuted_trs <- get_permuted_order(in_y, permute_params$n_cycles)
    # only shuffle the y order (the real BOLD timepoints in the TESTING data)
    df_xval %<>%
      mutate(obs_permuted = map(preds, \(x) x %>% 
                                  select(run_type, run_num, stim_type, tr_num, obs) %>% 
                                  # bind on the real-to-permuted TR mappings
                                  # this SHOULD also flexibly handle when the data are split up by run type
                                  # because left_join prioritizes the rows in the (subset) xval data
                                  left_join(permute_order,
                                            by = c("run_type",
                                                   "run_num",
                                                   "stim_type",
                                                   "tr_num")) %>%
                                  # drop the real TRs
                                  select(-tr_num) %>% 
                                  rename(tr_num = tr_num_new)
      ),
      # drop the real obs and paste the permuted obs back on
      preds = map2(preds, obs_permuted, \(x, y) x %>% 
                     select(-obs) %>% 
                     left_join(y,
                               by = c("run_type",
                                      "run_num",
                                      "stim_type",
                                      "tr_num")))) %>% 
      select(-obs_permuted)
  }
  
  out <- df_xval %>% 
    # doing groupavg separately for each xval fold should allow us to exclude
    # the held-out subject from each group-average timeseries
    left_join(in_groupavg) %>% 
    mutate(perf = map2(preds, groupavg,
                      \(x, y) calc_perf(x, groupavg = y), 
                      .progress = "Estimating encoding performance")
    ) %>% 
    select(-groupavg)
  
  if (decoding) {
    out %<>%
      mutate(decoding = map(preds,
                            \(x) get_decoding(x, 
                                              model_spec = parsnip::pls(mode = "classification", 
                                                                        predictor_prop = 1, 
                                                                        num_comp = 10L)),
                            .progress = "Fitting decoding model"
             ))
  }
  
  return (out)
}

calc_perf <- function (pred, groupavg) {
  # calculate q-squared and r-squared
  if ("run_type" %in% names(groupavg)) {
    out <- pred %>% 
      left_join(groupavg, by = c("run_type", "run_num", "stim_type", "tr_num", "voxel_num"))
  } else {
    out <- pred %>% 
      left_join(groupavg, by = c("run_num", "stim_type", "tr_num", "voxel_num"))
  }
  out %<>%
    # A SEPARATE METRIC FOR EACH SUBJECT X VOXEL
    # For the one-value-per-subject ones, AVERAGE ACROSS VOXELS
    # Similarly, for the one-value-per-subject ones, AVERAGE ACROSS SUBJECTS
    # Basically, deal with it LATER not HERE
    rename(model = pred) %>% 
    group_by(stim_type, subj_num, voxel_num) %>% 
    summarize(tss = sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE),
              # Apparently you can reference other non-across columns inside across() anon functions, snazzy
              across(c(model, groupavg),
                     list(rss = \(x) sum((obs-x)^2, na.rm = TRUE),
                          r = \(x) cor(x, obs, use = "complete.obs")),
                     .names = "{.fn}_{.col}"),
              .groups = "drop") %>% 
    # calculate q2 in place and then rename
    mutate(across(starts_with("rss"), \(x) (1 - (x/tss)))) %>%
    rename_with(\(x) str_replace(x, "rss", "q2"), .cols = starts_with("rss")) %>% 
    select(-tss)
  
  return (out)
}

# External function to calculate the group average timeseries from the original y
# Do not calculate this from the true values bundled along with the predictions anymore
# because the fit/predicted values from the training subjects are getting thrown out
# to make the pred object smaller
calc_groupavg_timeseries <- function (in_y) {
  out <- in_y %>% 
    group_by(run_type, run_num, stim_type, tr_num) %>%
    summarize(across(starts_with("voxel"), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>% 
    pivot_longer(cols = starts_with("voxel"),
                 names_to = "voxel_num",
                 values_to = "groupavg",
                 names_prefix = "voxel_",
                 names_transform = list(voxel_num = as.integer))
  
  return (out)
}

# This function currently does nothing
compare_varexp <- function (pred) {
  # fit the variance explained model between the two pls-predicted timecourses
  out <- pred %>% 
    # normalize the predicted timecourses before checking variance explained stuff
    # so that the the lm betas are on the same scale
    mutate(across(starts_with("pred_"), \(x) c(scale(x)))) %>% 
    lm(obs ~ pred_flynet + pred_prf, data = .) %>% 
    tidy() %>% 
    select(term, estimate) %>% 
    filter(term != "(Intercept)")
  
  return (out)
}

get_decoding <- function (pred, model_spec, stim_type_to_classify = "ring_expand") {
  pred %<>% 
    pivot_longer(cols = c(pred, obs), names_to = "data_type", values_to = "bold") %>% 
    pivot_wider(names_from = voxel_num,
                values_from = bold,
                names_prefix = "voxel_")
  
  train_data <- pred %>% 
    filter(data_type == "pred")
  
  test_data <- pred %>%
    filter(data_type == "obs")
  
  decoder_recipe <- recipe(head(train_data)) %>% 
    update_role(starts_with("voxel"), new_role = "predictor") %>% 
    # Hacking these functions to create a binary factor variable for only one level of stim_type >:)
    step_dummy_extract(stim_type,
                       role = "outcome",
                       pattern = stim_type_to_classify,
                       other = NULL,
                       # this names the dummy variable columns just with the condition names
                       naming = \(var, lvl, ordered = FALSE) lvl,
                       # skip = TRUE for these two because they operate on outcome vars
                       skip = TRUE) %>% 
    step_bin2factor(all_outcomes(), skip = TRUE) %>% 
    update_role_requirements(role = NA, bake = FALSE)
  
  decoder_workflow <- workflow() %>% 
    # mixOmics is the only available engine so just leave it
    add_model(model_spec)
  
  # get and process model preds
  decoder_fit <- decoder_workflow %>% 
    add_recipe(decoder_recipe) %>% 
    fit(data = train_data)
  
  out_train <- decoder_fit %>% 
    # look b I just want the model fits
    predict(new_data = train_data) %>% 
    bind_cols(decoder_recipe %>% 
                prep(train_data) %>% 
                bake(new_data = NULL, all_outcomes())) %>% 
    rename(true_class = !!stim_type_to_classify, pred_class = .pred_class) %>% 
    mutate(across(ends_with("_class"), \(x) factor(x))) %>% 
    metrics(truth = true_class, estimate = pred_class)
  
  out_test <- decoder_fit %>% 
    predict(new_data = test_data) %>% 
    bind_cols(decoder_recipe %>% 
                prep(test_data) %>% 
                bake(new_data = NULL, all_outcomes())) %>% 
    rename(true_class = !!stim_type_to_classify, pred_class = .pred_class) %>% 
    mutate(across(ends_with("_class"), \(x) factor(x))) %>% 
    metrics(truth = true_class, estimate = pred_class)
  
  out <- bind_rows(pred = out_train, obs = out_test, .id = "decoding_type")
  
  return (out)
  
}

## putting it all together ----

# The DEFAULT is leave-one-subject-out
prep_xval <- function (in_y, n_folds = NULL, by_run_type = FALSE) {

  n_subjs <- length(unique(in_y$subj_num))
  if (is.null(n_folds)) n_folds <- n_subjs
  
  if (by_run_type) {
    out <- crossing(fold_num = 1:n_folds,
                    this_run_type = unique(in_y$run_type))
  } else {
    out <- tibble(fold_num = 1:n_folds)
  }
  
  out %<>%
    mutate(test_subjs = map(fold_num, \(x) as.integer(seq(x, n_subjs, n_folds))))
  
  return (out)
}

fit_xval <- function (in_x, in_y, n_folds = NULL, by_run_type = FALSE, include_fit = TRUE) {
  out <- prep_xval(in_y, n_folds, by_run_type)
  
  if (by_run_type) {
    out %<>%
      mutate(preds = map2(test_subjs, this_run_type,
                         \(x, y) get_pls_preds(in_x = in_x %>% 
                                              filter(run_type == y),
                                            in_y = in_y %>% 
                                              filter(run_type == y),
                                            test_subjs = x,
                                            include_fit = include_fit),
                         .progress = "Estimating one xval round"))
  } else {
    out %<>%
      mutate(preds = map(test_subjs,
                         \(x) get_pls_preds(in_x = in_x,
                                            in_y = in_y,
                                            test_subjs = x,
                                            include_fit = include_fit),
                         .progress = "Estimating one xval round"))
  }
  
  if (include_fit) {
    out %<>%
      # for some reason, unnest_wider doesn't play nice with list-cols of workflow objects
      # so need to hoist manually twice
      hoist(preds, "fit") %>% 
      rename(fits = fit)
  }
  out %<>%
    hoist(preds, "pred") %>% 
    rename(preds = pred)
  
  return (out)
}

