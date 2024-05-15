## helper functions for constructing SPM calls used by targets ----

# realign and estimate studyforrest BOLD data
spm_realign.norm_studyforrest <- function (bold_wedge.counter,
                                           bold_wedge.clock,
                                           bold_ring.contract,
                                           bold_ring.expand,
                                           script, 
                                           n_trs = 90,
                                           out_prefix = "w") {
  bold_all <- c(bold_wedge.counter, bold_wedge.clock, bold_ring.contract, bold_ring.expand)
  out_paths <- file.path(dirname(bold_all), paste0(out_prefix, basename(bold_all)))
  
  matlab_commands = c(
    rvec_to_matlabcell(
      paste(bold_wedge.counter, 
            1:n_trs, 
            sep = ","),
      matname = "paths_nifti_ccw"),
    rvec_to_matlabcell(
      paste(bold_wedge.clock, 
            1:n_trs, 
            sep = ","),
      matname = "paths_nifti_clw"),
    rvec_to_matlabcell(
      paste(bold_ring.contract, 
            1:n_trs, 
            sep = ","),
      matname = "paths_nifti_con"),
    rvec_to_matlabcell(
      paste(bold_ring.expand, 
            1:n_trs, 
            sep = ","),
      matname = "paths_nifti_exp"),
    assign_variable("out_prefix", out_prefix),
    call_script(script)
  )
  
  with_path(
    matlab_path, # assume this is a global variable that will be instantiated in the targets script
    run_matlab_code(matlab_commands)
  )
  
  return (out_paths)
}