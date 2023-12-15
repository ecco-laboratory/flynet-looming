## setup ----

# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

## modeling and helper functions for said ----

workflow_eyeblink <- function (in_data, pca_vars = c("hit_prob", "tau_inv", "eta"), return_fit = FALSE) {
  out <- workflow() %>% 
    add_model(poisson_reg()) %>% 
    add_recipe(in_data %>% 
                 recipe_eyeblink(pca_vars = pca_vars))
  
  if (return_fit) {
    out <- out %>% 
      fit(data = in_data) %>% 
      extract_fit_engine()
  }
  
  return (out) 
}

recipe_eyeblink <- function (in_data, pca_vars) {
  out_recipe <- in_data %>% 
    # filter(frame_num_shifted >= 125) %>% 
    # select(-starts_with("frame_num")) %>% 
    recipe() %>% 
    update_role(n_blinks, new_role = "outcome") %>% 
    update_role(c("ttc", !!pca_vars), new_role = "predictor") %>% 
    # to get an equal number of frames per ttc (from the end of the video)
    step_filter(frame_num_shifted >= 125) %>% 
    # step_rm(has_role(NA)) %>% 
    # for more interpretable coefs
    # hit prob was previously being multiplied by 10
    # but now it's getting PCA'd so that wouldn't do anything anymore
    step_range(ttc, min = 0, max = 4)
  
  if (length(pca_vars) > 1) {
    out_recipe %<>%
      step_pca(all_of(!!pca_vars), 
               num_comp = length(pca_vars), 
               options = list(center = TRUE, scale. = TRUE))
  } else {
    out_recipe %<>%
      step_normalize(all_of(!!pca_vars))
  }
    
  return (out_recipe)
}

## permuting ----

# just re-fitting a pure glm is fast enough
# that we don't need to break these up across jobs... FOR NOW

# Note to self from Phil 06/30/2023: The goal with his push for the thresholded quartile analyses
# is to characterize something about the discreteness/thresholdiness of the hit prob/blink process

perm_eyeblink <- function (in_data, pca_vars, times) {
  this_workflow <- workflow_eyeblink(in_data,
                                     pca_vars = pca_vars)
  
  out <- in_data %>% 
    permutations(permute = n_blinks, times = times) %>% 
    mutate(coefs = map(splits, \(x) fit(this_workflow, analysis(x)) %>% 
                         tidy(),
                       .progress = "Permuting for Poisson coef")) %>% 
    select(-splits) %>% 
    unnest(coefs)
  
  return (out)
}
