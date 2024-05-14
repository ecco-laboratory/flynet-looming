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
  out <- read_csv(file, 
                  col_names = c("frame_num", "filename", paste0("unit_", 1:256)),
                  # bc we're applying new col names
                  skip = 1) %>% 
    # note: the first flow-frame is anchored to the second real frame
    # but do not add the +1 frame_num adjustment 
    # so that the last frame doesn't start rounding up to a TR past the stimulus
    mutate(frame_num = (1:n()), 
           # int division starts TR count at 0, bump up to 1
           # plus whatever TR offset if the scan starts before the video plays
           tr_num = as.integer(frame_num %/% (fps * tr_length) + tr_start_offset + 1)) %>% 
    select(filename, frame_num, tr_num, everything()) %>% 
    pivot_longer(cols = starts_with("unit"),
                 names_to = "rf",
                 values_to = "activation",
                 names_prefix = "unit_",
                 names_transform = list(rf = as.integer)) %>% 
    nest(activations = -c(filename, tr_num, rf)) %>% 
    mutate(intercept = map_dbl(activations,
                       ~lm(activation ~ scale(frame_num, scale = FALSE), data = .) %>% 
                         # DROP SLOPES BECAUSE WE AREN'T ANALYZING THEM IN THIS THING AT PRESENT
                         pluck("coefficients", "(Intercept)"),
                       .progress = "estimating RF activation intercepts")) %>% 
    select(-activations) %>% 
    pivot_wider(id_cols = c(filename, tr_num),
                names_from = rf,
                values_from = intercept,
                # use a more agnostic name because this is going to be the general prefix for any predictor
                names_prefix = "unit_")
    
  return (out)
}

# targets tracks external files as their paths so the paths must be the input
# extract stimulus conditions from the file paths
get_flynet_activation_studyforrest <- function (file) {
  out <- get_flynet_activation(file = file,
                               fps = 25,
                               tr_length = 2,
                               # first two TRs in the scan are rest
                               # aka the video wasn't playing
                               tr_start_offset = 2) %>% 
    # get the filename (without folders and without file extension) as the run name
    # which in this case is also stimulus type
    mutate(run_type = tools::file_path_sans_ext(filename)) %>% 
    select(-filename)
  
  return (out)
}
