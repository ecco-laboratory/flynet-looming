## setup ----

# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

## modeling and helper functions for said ----

premodel_eyeblink <- function (in_data) {
  out <- in_data %>% 
    # to get an equal number of frames per ttc (from the end of the video)
    filter(frame_num_shifted >= 125) %>% 
    # for more interpretable coefs
    mutate(hit_prob = hit_prob * 10,
           ttc = ttc - 3)
  
  return (out)
}

## permuting ----

# just re-fitting a pure glm is fast enough
# that we don't need to break these up across jobs... FOR NOW

# Note to self from Phil 06/30/2023: The goal with his push for the thresholded quartile analyses
# is to characterize something about the discreteness/thresholdiness of the hit prob/blink process

perm_eyeblink_hit.prob <- function (in_data, times) {
  in_data %>% 
    permutations(permute = n_blinks, times = times) %>% 
    mutate(coefs = map(splits, \(x) x %>% 
                         analysis() %>% 
                         glm(n_blinks ~ hit_prob, family = "poisson", data = .) %>% 
                         tidy(),
                       .progress = "Permuting for hit prob coef")) %>% 
    select(-splits) %>% 
    unnest(coefs)
}

perm_eyeblink_ttc <- function (in_data, times) {
  in_data %>% 
    nest(data = -ttc) %>% 
    permutations(permute = ttc, times = times) %>% 
    mutate(coefs = map(splits, \(x) x %>% 
                         analysis() %>% 
                         unnest(data) %>% 
                         glm(n_blinks ~ ttc, family = "poisson", data = .) %>% 
                         tidy(),
                       .progress = "Permuting for TTC coef")) %>% 
    select(-splits) %>% 
    unnest(coefs)
}