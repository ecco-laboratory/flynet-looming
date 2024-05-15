## helper functions for indexing and gunzipping nifti.gz used by targets ----

search_niftis_gz <- function (subj_id,
                              task_id,
                              fmri_dir = here::here("ignore", "datasets", "studyforrest-data-phase2")) {
  regex_pattern = paste0(".*", task_id, "_run-1_bold.nii.gz")
  niftis_gz <- list.files(file.path(fmri_dir, subj_id), 
                          pattern = regex_pattern, 
                          recursive = TRUE,
                          full.names = TRUE)
  return (niftis_gz)
}
gunzip_nifti <- function (nifti_gz) {
  system2("gunzip",
          args = c("-kf",
                   nifti_gz))
  
  nifti <- str_remove(nifti_gz, ".gz")
  return (nifti)
}