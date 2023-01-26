## setup ----

require(mixOmics)
require(tidymodels)
require(plsmod)
# load tidyverse after all that stuff because there are functions in the above packages that mask select() and map()
require(tidyverse)
require(magrittr)
require(crayon)

skip_v1 <- TRUE
n_perms <- 500
skip_perms <- FALSE
in_parallel <- TRUE

if (in_parallel) {
  # Not trying to blow up node1
  n_cores <- 16
  options(mc.cores = n_cores)
  future::plan("multicore", workers = n_cores)
  cat("Will run permutations in parallel with " %+% green(n_cores) %+% " cores", fill = TRUE)
}

# using the parameters to reproduce the SPM default HRF
# SPM uses a 32-s kernel, mind thee
double_gamma_hrf <- function (time, shape1 = 6, shape2 = 16, scale1 = 1, scale2 = 1, scale_undershoot = 1/6) {
  gamma1 <- dgamma(time, shape = shape1, scale = scale1)
  gamma2 <- scale_undershoot * dgamma(time, shape = shape2, scale = scale2)
  
  return (gamma1 - gamma2)
}

convolve_hrf <- function (input, kernel_length = 32, tr = 2) {
  out <- convolve(input, rev(double_gamma_hrf(seq(0, kernel_length, tr))), type = "open")
  # trim back down to the original timecourse length
  out <- out[1:length(input)]
  return (out)
}

## get flynet activation timecourses ----

activation_ints_convolved <- tibble(stim_type = c("ring_expand", "ring_contract", "wedge_clock", "wedge_counter")) %>% 
  mutate(data = map(stim_type,
                    ~read_csv(
                      here::here("ignore",
                                 "outputs",
                                 paste0("flynet_132x132_stride8_activations_studyforrest_retinotopy_",
                                        .x,
                                        ".csv")),
                      col_names = paste0("unit_", 1:256)))) %>% 
  unnest(data) %>% 
  group_by(stim_type) %>% 
  # adjust flow-frame up by 1 to account for the fact
  # that flow can only be calculated starting on the second actual frame
  mutate(frame_num = (1:n()) + 1L,
         # frames are 25 fps, so TRs are 50 frames
         # int division starts TR count at 0, bump up to 3
         # because the first 2 TRs are rest, per the data paper
         tr_num = frame_num %/% 50L + 3L) %>% 
  ungroup() %>% 
  select(stim_type, frame_num, tr_num, everything()) %>% 
  pivot_longer(cols = starts_with("unit"),
               names_to = "rf",
               values_to = "activation",
               names_prefix = "unit_",
               names_transform = list(rf = as.integer)) %>% 
  nest(activations = -c(stim_type, tr_num, rf)) %>% 
  mutate(coefs = map(activations,
                     ~lm(activation ~ scale(frame_num, scale = FALSE), data = .) %>% 
                       pluck("coefficients"),
                     .progress = "RF activation slopes")) %>% 
  select(-activations) %>% 
  unnest_wider(coefs) %>% 
  rename(intercept = "(Intercept)", slope = "scale(frame_num, scale = FALSE)") %>% 
  # DROP SLOPES BECAUSE WE AREN'T ANALYZING THEM IN THIS THING AT PRESENT
  select(-slope) %>% 
  pivot_wider(id_cols = c(stim_type, tr_num),
              names_from = rf,
              values_from = intercept,
              names_prefix = "intercept_") %>% 
  group_by(stim_type) %>% 
  # tidymodels step_convolve was proposed and then... cancelled by requester?
  # so we have to do this before pushing through recipes
  mutate(across(starts_with("intercept"), \(x) c(scale(convolve_hrf(x))))) %>% 
  ungroup()

write_rds(activation_ints_convolved, here::here("ignore", "outputs", "studyforrest_retinotopy_flynet_timecourses.rds"))

## get fmri data ----
cat(magenta("Reading in brain data"), fill = TRUE)

# subject by voxel by timepoint by condition
# note: Matlab by default, and thus Phil’s CANLabTools,
# repeats fastest along x, then y, then z when making mask voxels and other arrays long
get_phil_matlab_fmri_data <- function (file, region, tr_start = NULL, tr_end = NULL, verbose = FALSE) {
  stim_types <- c("wedge_counter", "wedge_clock", "ring_contract", "ring_expand")
  data_mat <- R.matlab::readMat(file, verbose = verbose)
  
  # if start and end TRs to filter are not specified,
  # default to all the TRs
  if (is.null(tr_start)) tr_start <- 1
  if (is.null(tr_end)) tr_end <- dim(data_mat$DATA)[3]
  
  # crossing() repeats the last columns fastest
  # so specify the dim-columns in reverse order from how they appear in the matlab data
  # R also repeats fastest along the first dim when representing arrays as vectors
  data <- crossing(stim_type = stim_types,
                      tr_num = 1:dim(data_mat$DATA)[3],
                      voxel = 1:dim(data_mat$DATA)[2],
                      subj_num = 1:dim(data_mat$DATA)[1]) %>%
    mutate(bold = as.vector(data_mat$DATA)) %>%
    pivot_wider(id_cols = c(subj_num, stim_type, tr_num),
                names_from = voxel,
                values_from = bold,
                names_prefix = paste0(region, "_")) %>% 
    # in the studyforrest data, the first 2 and last 8 TRs are rest
    # it will be better later for the pRF predicted data if we remove the rest TRs now
    filter(tr_num >= tr_start, tr_num <= tr_end) %>% 
    group_by(stim_type, subj_num) %>% 
    mutate(across(starts_with(region), \(x) c(scale(x)))) %>% 
    ungroup()
  
  return (data)
}

if (!skip_v1) {
  v1_data <- get_phil_matlab_fmri_data("/home/data/eccolab/studyforrest-data-phase2/V1_DATA_bpf.mat",
                                       region = "v1",
                                       tr_start = 3,
                                       tr_end = 82) %>% 
    # Just this subject has all 0 data for some reason...
    filter(subj_num != 6) 
}

sc_data <- get_phil_matlab_fmri_data("/home/data/eccolab/studyforrest-data-phase2/DATA_bpf.mat",
                                     region = "sc",
                                     tr_start = 3,
                                     tr_end = 82)

mask_nifti <- RNifti::readNifti("/home/data/eccolab/studyforrest-data-phase2/SC_mask_vox_indx.nii")

# which() repeats the x coords fastest, then y, then z
# which should be the same as what matlab is doing 
# when Phil's code collapses the voxels to 1d
mask_voxel_coords <- which(mask_nifti != 0, arr.ind = TRUE) %>% 
  as_tibble() %>% 
  rename(x = dim1, y = dim2, z = dim3) %>% 
  mutate(voxel_num = 1:n()) %>% 
  mutate(side = if_else(x <= 39, "right", "left"))

## get prf estimated timecourses ----

# These should now be estimated only for one "subject" per FOLD, but across five FOLDS
# because it takes the stimuli and one set of params that was tuned on the super-subj
# aka all 12 training subjs concatenated


sc_pred_prf_xval <- get_phil_matlab_fmri_data('/home/data/eccolab/studyforrest-data-phase2/pred_sc_prf_xval.mat',
                                              region = "prf") %>% 
  rename(fold_num = subj_num) %>% 
  # since these were generated without the rest TRs
  mutate(tr_num = tr_num + 2L) %>% 
  nest(prfs = -c(stim_type, fold_num))

n_folds <- max(sc_pred_prf_xval$fold_num)

if (!skip_v1) {
  v1_pred_prf_xval <- get_phil_matlab_fmri_data('/home/data/eccolab/studyforrest-data-phase2/pred_v1_prf_xval.mat',
                                                region = "prf") %>% 
    rename(fold_num = subj_num) %>% 
    mutate(tr_num = tr_num + 2L) %>% 
    nest(prfs = -c(stim_type, fold_num))
}

## bind predictors and outcomes together ----
cat(magenta("Making split data for modeling"), fill = TRUE)

split_data_sc <- sc_data %>% 
  left_join(activation_ints_convolved,
            by = c("stim_type", "tr_num")) %>% 
  nest(data = -stim_type) %>% 
  slice(rep(1:n(), times = n_folds)) %>% 
  mutate(fold_num = rep(1:n_folds, each = length(unique(sc_data$stim_type)))) %>% 
  mutate(test_subjs = map(fold_num, \(x) seq(x, length(unique(sc_data$subj_num)), n_folds))) %>% 
  left_join(sc_pred_prf_xval, by = c("stim_type", "fold_num")) %>% 
  mutate(data = map2(data, prfs, \(x, y) left_join(x, y, by = "tr_num"))) %>% 
  select(-prfs) %>% 
  unnest(data) %>% 
  # Need to manually separate subjs
  # so it's the same separation as the pRF-trained subjs in the matlab script
  mutate(split_type = map2_chr(subj_num, test_subjs, 
                               \(x, y) {
                                 if (x %in% y) {
                                   return ("test")
                                 } else {
                                   return ("train")
                                 }
                               })) %>% 
  select(-test_subjs) %>% 
  select(stim_type, split_type, subj_num, tr_num, everything()) %>% 
  nest(data = -c(stim_type, fold_num))

if (!skip_v1) {
  split_data_v1 <- v1_data %>% 
    left_join(activation_ints_convolved,
              by = c("stim_type", "tr_num")) %>% 
    nest(data = -stim_type) %>% 
    slice(rep(1:n(), times = n_folds)) %>% 
    mutate(fold_num = rep(1:n_folds, each = length(unique(v1_data$stim_type)))) %>% 
    mutate(test_subjs = map(fold_num, \(x) seq(x, length(unique(v1_data$subj_num)), n_folds))) %>% 
    left_join(v1_pred_prf_xval, by = c("stim_type", "fold_num")) %>% 
    mutate(data = map2(data, prfs, \(x, y) left_join(x, y, by = "tr_num"))) %>% 
    select(-prfs) %>% 
    unnest(data) %>% 
    # Need to manually separate subjs
    # so it's the same separation as the pRF-trained subjs in the matlab script
    mutate(split_type = map2_chr(subj_num, test_subjs, 
                                 \(x, y) {
                                   if (x %in% y) {
                                     return ("test")
                                   } else {
                                     return ("train")
                                   }
                                 })) %>% 
    select(-test_subjs) %>% 
    select(stim_type, split_type, subj_num, tr_num, everything()) %>% 
    nest(data = -c(stim_type, fold_num))
}

## tidymodels setup ----

base_sc_recipe <- recipe(split_data_sc %>%
                        pull(data) %>% 
                        pluck(1) %>% 
                        head()) %>% 
  update_role(starts_with("sc"), new_role = "outcome") %>% 
  update_role(subj_num, new_role = "ID")

flynet_sc_recipe <- base_sc_recipe %>% 
  update_role(starts_with("intercept"), new_role = "predictor")

prf_sc_recipe <- base_sc_recipe %>% 
  update_role(starts_with("prf"), new_role = "predictor")

if (!skip_v1) {
  base_v1_recipe <- recipe(split_data_v1 %>%
                             pull(data) %>% 
                             pluck(1) %>% 
                             head()) %>% 
    update_role(starts_with("v1"), new_role = "outcome") %>% 
    update_role(subj_num, new_role = "ID")
  
  flynet_v1_recipe <- base_v1_recipe %>% 
    update_role(starts_with("intercept"), new_role = "predictor")
  
  prf_v1_recipe <- base_v1_recipe %>% 
    update_role(starts_with("prf"), new_role = "predictor")
}

pls_workflow <- workflow() %>% 
  # mixOmics is the only available engine so just leave it
  add_model(parsnip::pls(mode = "regression",
                         predictor_prop = 0.5,
                         num_comp = 3L))

## actually fit shit ----

# Returns a df with ONLY subj_num, old (true) TR, and permuted TR
# to be used for feeding into get_pls_metrics() so that the same randomization
# can be applied across different xval folds with the same iteration number
get_permuted_order <- function (in_data) {
  permuted_trs <- in_data %>% 
    select(subj_num, tr_num) %>% 
    # this will make it easier to permute within each of the 5 stimulus cycles
    mutate(cycle_num = (tr_num - 3L) %/% 16L) %>% 
    group_by(subj_num, cycle_num) %>% 
    slice_sample(prop = 1) %>% 
    mutate(tr_num_new = 1:n() + (16L * cycle_num) + 2L) %>% 
    ungroup() %>% 
    select(-cycle_num)
  
  return (permuted_trs)
}

# Function written to wrap all the metrics calculations for a single iteration of data
# in preparation for running this across a horrifying permutation test
get_pls_metrics <- function (in_data, region = "sc", permute_order = NULL, return_preds = FALSE) {
  
  if (!is.null(permute_order)) {
    in_predictors <- in_data %>% 
      select(-starts_with(region))
    
    in_outcomes <- in_data %>% 
      select(-starts_with("intercept"), -starts_with("prf")) %>% 
      # bind on the real-to-permuted TR mappings
      left_join(permute_order, by = c("subj_num", "tr_num")) %>% 
      # drop the real TRs
      select(-tr_num) %>% 
      rename(tr_num = tr_num_new)
    
    # bind with the real TRs of the predictors and the permuted TRs of the outcomes
    in_data <- full_join(in_predictors, in_outcomes, by = c("split_type", "subj_num", "tr_num"))
  }
  
  train_data <- in_data %>% 
    filter(split_type == "train")
  
  if (region == "sc") {
    flynet_recipe <- flynet_sc_recipe
    prf_recipe <- prf_sc_recipe
  } else if (region == "v1") {
    flynet_recipe <- flynet_v1_recipe
    prf_recipe <- prf_v1_recipe
  }
  # get model preds
  pred_flynet <- pls_workflow %>% 
    add_recipe(flynet_recipe) %>% 
    fit(data = train_data) %>% 
    predict(new_data = in_data) %>% 
    bind_cols(in_data, .) %>% 
    select(-starts_with("intercept"), -starts_with("prf"))
  
  pred_prf <- pls_workflow %>% 
    add_recipe(prf_recipe) %>% 
    fit(data = train_data) %>% 
    predict(new_data = in_data) %>% 
    bind_cols(in_data, .) %>% 
    select(-starts_with("intercept"), -starts_with("prf"))
  
  # bind and process model preds
  pred <- bind_rows(flynet = pred_flynet,
                    prf = pred_prf,
                    .id = "encoding_type") %>% 
    rename_with(.fn = \(x) paste0("obs_", x), .cols = starts_with(region)) %>% 
    pivot_longer(cols = contains(paste0("_", region, "_")), 
                 names_to = c(".value", "voxel_num"), 
                 names_pattern = paste0("(.*)_", region, "_(.*)"),
                 names_transform = list(voxel_num = as.integer)) %>% 
    pivot_wider(names_from = encoding_type,
                values_from = .pred)
  
  # fit the variance explained model between the two pls-predicted timecourses
  var_explained <- pred %>% 
    filter(split_type == "test") %>% 
    select(-split_type) %>% 
    # normalize the predicted timecourses before checking variance explained stuff
    # so that the the lm betas are on the same scale
    mutate(across(c(flynet, prf), \(x) c(scale(x)))) %>% 
    lm(obs ~ flynet + prf, data = .) %>% 
    tidy() %>% 
    select(term, estimate) %>% 
    filter(term != "(Intercept)")
  
  # calculate q-squared and r-squared
  perf <- pred %>% 
    # NOT grouping by split_type bc groupavg should be taken across the train and test subjs
    group_by(voxel_num, tr_num) %>% 
    mutate(groupavg = mean(obs, na.rm = TRUE),
           across(c(flynet, prf, groupavg), \(x) {x - obs}, .names = "error_{.col}")) %>% 
    select(-c(flynet, prf, groupavg)) %>% 
    pivot_longer(cols = starts_with("error"),
                 names_to = "encoding_type",
                 values_to = "error",
                 names_prefix = "error_") %>% 
    group_by(split_type, encoding_type) %>% 
    summarize(tss = sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE),
              rss = sum(error^2, na.rm = TRUE),
              .groups = "drop") %>% 
    mutate(metric = 1 - (rss/tss)) %>% 
    select(-tss, -rss) %>% 
    pivot_wider(names_from = split_type,
                values_from = metric) %>% 
    rename(q2 = test, r2 = train)
  
  # for subject-wise plotskies
  perf_subj <- pred %>% 
    filter(split_type == "test") %>% 
    # NOT grouping by encoding_type bc groupavg should be taken across the train and test subjs
    mutate(across(c(flynet, prf), \(x) {x - obs}, .names = "error_{.col}")) %>% 
    select(-flynet, -prf) %>% 
    pivot_longer(cols = starts_with("error"),
                 names_to = "encoding_type",
                 values_to = "error",
                 names_prefix = "error_") %>% 
    group_by(encoding_type, subj_num) %>% 
    summarize(tss = sum((obs - mean(obs))^2),
              rss = sum(error^2),
              .groups = "drop") %>% 
    mutate(q2 = 1 - (rss/tss)) %>% 
    select(-tss, -rss)
  
  # for brain maps
  # each voxel gets a q-squared value--calculate it across TRs and test subjs
  perf_voxel <- pred %>% 
    filter(split_type == "test") %>% 
    # NOT grouping by encoding_type bc groupavg should be taken across the train and test subjs
    mutate(across(c(flynet, prf), \(x) {x - obs}, .names = "error_{.col}")) %>% 
    select(-flynet, -prf) %>% 
    pivot_longer(cols = starts_with("error"),
                 names_to = "encoding_type",
                 values_to = "error",
                 names_prefix = "error_") %>% 
    group_by(encoding_type, voxel_num) %>% 
    summarize(tss = sum((obs - mean(obs))^2),
              rss = sum(error^2),
              .groups = "drop") %>% 
    mutate(q2 = 1 - (rss/tss)) %>% 
    select(-tss, -rss)
  
  # return_preds is FALSE by default so the output is smaller
  if (return_preds) {
    out <- list(pred = pred,
                varexp = var_explained,
                perf_overall = perf,
                perf_subj = perf_subj,
                perf_voxel = perf_voxel)
  } else {
    out <- list(varexp = var_explained,
                perf_overall = perf,
                perf_subj = perf_subj,
                perf_voxel = perf_voxel)
  }
}

metrics_sc <- split_data_sc %>% 
  mutate(metric = map(data, \(x) get_pls_metrics(x, return_preds = TRUE), .progress = "sc model fittin n metric calc")) %>% 
  select(-data) %>% 
  unnest_wider(metric)

write_rds(metrics_sc, here::here("ignore", "outputs", "studyforrest_retinotopy_sc_pls_metrics.rds"))

if (!skip_v1) {
  # Lol ugh it's fucking huge
  metrics_v1 <- split_data_v1 %>% 
    mutate(metric = map(data,
                        \(x) get_pls_metrics(x, region = "v1", return_preds = TRUE), 
                        .progress = "v1 model fittin n metric calc")) %>% 
    select(-data) %>% 
    unnest_wider(metric)
  
  write_rds(metrics_v1, here::here("ignore", "outputs", "studyforrest_retinotopy_v1_pls_metrics.rds"))
} else {
  cat(red("Skipping V1 metrics for time"), fill = TRUE)
}

if (!skip_perms) {
  cat("About to execute " %+% magenta$bold(n_perms) %+% " full permutation runs, ugh!", fill = TRUE)
  # cat(red("Only for expanding rings right now!!!"), fill = TRUE)
  
  metrics_sc_perm <- split_data_sc %>% 
    select(stim_type, fold_num) %>% 
    # by nesting everything in, two stim conditions in the same iteration number
    # have the same permutation shuffles
    # idk that it matters that much but I feel better about
    # doing paired within-permutation statistics this way
    nest(params = everything()) %>% 
    # This is the number of permutation iterations!!!!
    slice(rep(1:n(), times = n_perms)) %>% 
    mutate(iteration = 1:n(),
           permute_order = map(iteration,
                               \(x) {
                                 # x is never actually called bc we just need to run 
                                 # the exact same code hella times
                                 split_data_sc %>% 
                                   # Fold num doesn't matter bc we're only operating over subj and TR IDs
                                   pull(data) %>% 
                                   pluck(1) %>% 
                                   get_permuted_order()
                               },
                               .progress = "Permuting TR order")) %>% 
    unnest(params)
  
  if (in_parallel) {
    cat("About to run " %+% red$bold(nrow(metrics_sc_perm)) %+% " total perms in parallel", fill = TRUE)
    metrics_sc_perm %<>% 
      mutate(metric = furrr::future_pmap(list(stim_type, fold_num, permute_order),
                                         \(stim, fold, perm) {
                                           split_data_sc %>% 
                                             filter(stim_type == stim, fold_num == fold) %>% 
                                             pull(data) %>% 
                                             pluck(1) %>% 
                                             get_pls_metrics(permute_order = perm)
                                         },
                                         .progress = TRUE))
    
  } else {
    metrics_sc_perm %<>% 
      mutate(metric = purrr::pmap(list(stim_type, fold_num, permute_order),
                                  \(stim, fold, perm) {
                                    split_data_sc %>% 
                                      filter(stim_type == stim, fold_num == fold) %>% 
                                      pull(data) %>% 
                                      pluck(1) %>% 
                                      get_pls_metrics(permute_order = perm)
                                  },
                                  .progress = "Permutation run in series :("))
  }
  metrics_sc_perm %<>% 
    unnest_wider(metric)
  
  write_rds(metrics_sc_perm, here::here("ignore", "outputs", "studyforrest_retinotopy_sc_pls_metrics_permuted.rds"))
} else {
  cat(red("Skipping permutation runs for time"), fill = TRUE)
}