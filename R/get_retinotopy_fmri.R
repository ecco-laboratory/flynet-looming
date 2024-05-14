## setup ----

# This is a targets-compatible function definition script
# Which means it should only be called under the hood by tar_make()
# and all the packages are loaded ELSEWHERE! Not in the body of this script.

## function definition ----

# note: Matlab by default, and thus Phil’s CANLabTools,
# repeats fastest along x, then y, then z when making mask voxels and other arrays long

get_phil_matlab_fmri_data_studyforrest <- function (file, verbose = TRUE) {
  
  data_mat <- R.matlab::readMat(file, verbose = verbose)
  
  # crossing() repeats the last columns fastest
  # so specify the dim-columns in reverse order from how they appear in the matlab data
  # R also repeats fastest along the first dim when representing arrays as vectors
  out <- crossing(run_type = 1:dim(data_mat$DATA)[4],
                  tr_num = 1:dim(data_mat$DATA)[3],
                  voxel = 1:dim(data_mat$DATA)[2],
                  subj_num = 1:dim(data_mat$DATA)[1]) %>% 
    # Jan 31 2023: HORROR OF HORRORS, crossing() reorders character vectors into ALPHABETICAL
    # before repeating. WHY!!!
    mutate(run_type = dplyr::recode(run_type,
                                  `1` = "wedge_counter",
                                  `2` = "wedge_clock",
                                  `3` = "ring_contract",
                                  `4` = "ring_expand")) %>% 
    # because only one run of each type was conducted
    # later code works better if every type of input data has this column, I suspect
    mutate(run_num = 1L)
  
  # OK. now ready to append the fucking data
  out %<>%
    mutate(bold = as.vector(data_mat$DATA))
  
  return (out)
}

proc_phil_matlab_fmri_data <- function (in_data, tr_start = NULL, tr_end = NULL) {
  # if start and end TRs to filter are not specified,
  # default to all the TRs
  if (is.null(tr_start)) tr_start <- 1
  if (is.null(tr_end)) tr_end <- max(in_data$tr_num)
  
  out <- in_data %>%
    pivot_wider(id_cols = c(subj_num, run_type, run_num, tr_num),
                names_from = voxel,
                values_from = bold,
                names_prefix = "voxel_") %>% 
    # in the studyforrest data, the first 2 and last 8 TRs are rest
    # it will be better later for the pRF predicted data if we remove the rest TRs now
    filter(tr_num >= tr_start, tr_num <= tr_end) %>% 
    group_by(run_type, run_num, subj_num) %>% 
    mutate(across(starts_with("voxel"), \(x) c(scale(x)))) %>% 
    ungroup()
  
  return (out)
}
