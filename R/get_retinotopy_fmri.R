## setup ----

require(tidyverse)
require(magrittr)

## function definition ----

get_phil_matlab_fmri_data_studyforrest <- function (file, verbose = TRUE) {
  
  data_mat <- R.matlab::readMat(file, verbose = verbose)
  
  # crossing() repeats the last columns fastest
  # so specify the dim-columns in reverse order from how they appear in the matlab data
  # R also repeats fastest along the first dim when representing arrays as vectors
  out <- crossing(run_id = 1:dim(data_mat$DATA)[4],
                  tr_num = 1:dim(data_mat$DATA)[3],
                  voxel = 1:dim(data_mat$DATA)[2],
                  subj_num = 1:dim(data_mat$DATA)[1]) %>% 
    # Jan 31 2023: HORROR OF HORRORS, crossing() reorders character vectors into ALPHABETICAL
    # before repeating. WHY!!!
    mutate(run_id = dplyr::recode(run_id,
                                  `1` = "wedge_counter",
                                  `2` = "wedge_clock",
                                  `3` = "ring_contract",
                                  `4` = "ring_expand"))
  
  # OK. now ready to append the fucking data
  out %<>%
    mutate(bold = as.vector(data_mat$DATA))
  
  return (out)
}

get_phil_matlab_fmri_data_nsd <- function (file, verbose = TRUE) {
  
  data_mat <- R.matlab::readMat(file, verbose = verbose)
  
  # bc the cols in crossing must be specified in reverse .mat dims order
  out <- crossing(voxel = 1:dim(data_mat$DATA)[4],
                  tr_num = 1:dim(data_mat$DATA)[3],
                  # this should reproduce the repeat structure described in NSD docs
                  run_id = 1:dim(data_mat$DATA)[2],
                  subj_num = 1:dim(data_mat$DATA)[1]) %>% 
    mutate(run_id = dplyr::recode(run_id,
                                  `1` = "bar_1",
                                  `2` = "wedgering_1",
                                  `3` = "floc_1",
                                  `4` = "floc_2",
                                  `5` = "bar_2",
                                  `6` = "wedgering_2",
                                  `7` = "floc_3",
                                  `8` = "floc_4",
                                  `9` = "bar_3",
                                  `10` = "wedgering_3",
                                  `11` = "floc_5",
                                  `12` = "floc_6"))
  
  out %<>%
    mutate(bold = as.vector(data_mat$DATA)) %>% 
    filter(!startsWith(run_id, "floc"))
  
  return (out)
}

proc_phil_matlab_fmri_data <- function (in_data, region, tr_start = NULL, tr_end = NULL) {
  # if start and end TRs to filter are not specified,
  # default to all the TRs
  if (is.null(tr_start)) tr_start <- 1
  if (is.null(tr_end)) tr_end <- max(in_data$tr_num)
  
  out <- in_data %>%
    pivot_wider(id_cols = c(subj_num, run_id, tr_num),
                names_from = voxel,
                values_from = bold,
                names_prefix = paste0(region, "_")) %>% 
    # in the studyforrest data, the first 2 and last 8 TRs are rest
    # it will be better later for the pRF predicted data if we remove the rest TRs now
    filter(tr_num >= tr_start, tr_num <= tr_end) %>% 
    group_by(run_id, subj_num) %>% 
    mutate(across(starts_with(region), \(x) c(scale(x)))) %>% 
    ungroup()
  
  return (out)
}
