## setup ----

# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

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

# In targets() world, this takes ONE file, because each file is what's tracked
get_flynet_activation <- function (file, fps, tr_length, tr_start_offset) {
  out <- read_csv(file, col_names = paste0("unit_", 1:256)) %>% 
    mutate(frame_num = (1:n()) + 1L, # + 1 because the first flow-frame is anchored to the second real frame
           # int division starts TR count at 0, bump up to 1
           # plus whatever TR offset if the scan starts before the video plays
           tr_num = as.integer(frame_num %/% (fps * tr_length) + tr_start_offset + 1)) %>% 
    select(frame_num, tr_num, everything()) %>% 
    pivot_longer(cols = starts_with("unit"),
                 names_to = "rf",
                 values_to = "activation",
                 names_prefix = "unit_",
                 names_transform = list(rf = as.integer)) %>% 
    nest(activations = -c(tr_num, rf)) %>% 
    mutate(intercept = map_dbl(activations,
                       ~lm(activation ~ scale(frame_num, scale = FALSE), data = .) %>% 
                         # DROP SLOPES BECAUSE WE AREN'T ANALYZING THEM IN THIS THING AT PRESENT
                         pluck("coefficients", "(Intercept)"),
                       .progress = "estimating RF activation intercepts")) %>% 
    select(-activations) %>% 
    pivot_wider(id_cols = tr_num,
                names_from = rf,
                values_from = intercept,
                # use a more agnostic name because this is going to be the general prefix for any predictor
                names_prefix = "unit_") %>% 
    # tidymodels step_convolve was proposed and then... cancelled by requester?
    # so we have to do this before pushing through recipes
    mutate(across(starts_with("unit"), \(x) c(scale(convolve_hrf(x)))))
    
  return (out)
}

# targets tracks external files as their paths so the paths must be the input
# extract stimulus conditions from the file paths
get_flynet_activation_studyforrest <- function (files) {
  out <- tibble(filename = files) %>% 
    mutate(data = map(filename,
                      \(x) get_flynet_activation(file = x,
                                                 fps = 25,
                                                 tr_length = 2,
                                                 # first two TRs in the scan are rest
                                                 # aka the video wasn't playing
                                                 tr_start_offset = 2
                      ))) %>% 
    unnest(data) %>% 
    # get the filename (without folders and without file extension) as the run name
    # which in this case is also stimulus type
    mutate(filename = str_split(filename, pattern = "[/.]"),
           run_type = map_chr(filename, \(x) x[length(x) - 1])) %>% 
    select(-filename)
  
  return (out)
}

get_flynet_activation_nsd <- function (files) {
  out <- tibble(filename = files) %>% 
    mutate(data = map(filename,
                      \(x) get_flynet_activation(file = x, fps = 15,
                                                 # the original data were acquired at tr = 1.6 s
                                                 # but this appears to have been run on the upsampled data
                                                 tr_length = 1,
                                                 tr_start_offset = 0
                      ))) %>% 
    unnest(data) %>% 
    mutate(filename = str_split(filename, pattern = "[/.]"),
           run_type = map_chr(filename, \(x) x[length(x) - 1])) %>% 
    select(-filename)
  # Leave run_id in there so we can bind this to the fMRI data by filename
  # but note that it will need suffixes because the stim videos are repeated in different runs
  
  return (out)
}
