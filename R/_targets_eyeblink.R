## setup ----

# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline # nolint

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes)
library(future)
library(future.callr)
library(future.batchtools)
library(tibble)
library(rlang)

# Set target options:
tar_option_set(
  packages = c("mixOmics",
               "tidymodels",
               "plsmod",
               "discrim",
               "tidyverse",
               "magrittr",
               "rlang"), # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
options(clustermq.scheduler = "slurm")
options(clustermq.template = "clustermq.tmpl")

# tar_make_future() configuration (okay to leave alone):
n_slurm_cpus <- 1L
plan(batchtools_slurm,
     template = "future.tmpl",
     resources = list(ntasks = 1L,
                      ncpus = n_slurm_cpus,
                      # nodelist = "node1",
                      # walltime 86400 for 24h (partition day-long)
                      # walltime 1800 for 30min (partition short)
                      walltime = 1800L,
                      # Please be mindful this is not THAT much. No honker dfs pls
                      memory = 250L,
                      partition = "short"))
# These parameters are relevant later inside the permutation testing targets
n_batches <- 50
n_reps_per_batch <- 200

# Run the R scripts in the R/ folder with your custom functions:
# tar_source(c())

## data files from other people's stuff ----

targets_raw_data <- list(
  tar_target(
    name = blink.counts_ayzenberg2015_raw,
    command = list.files("/home/mthieu/Repos/lourenco_ilooming",
                         full.names = TRUE),
    format = "file"
  )
)

targets_cleaned_data <- list(
  tar_target(
    blink.counts_ayzenberg2015,
    command = {
      tibble(filename = blink.counts_ayzenberg2015_raw) %>%
        mutate(data = map(filename, read_csv)) %>%
        unnest(data) %>% 
        rename(subj_num = SUBJ) %>%
        mutate(subj_num = as.integer(subj_num),
               ttc = as.integer(str_sub(filename, start = -8, end = -5))) %>%
        select(-filename, -starts_with("...")) %>%
        filter(!(subj_num %in% c(26, 38, 53:54))) %>%
        pivot_longer(cols = -c(subj_num, ttc),
                     names_to = "frame_num",
                     values_to = "n_blinks",
                     values_ptypes = list(n_blinks = integer()),
                     names_transform = list(frame_num = as.integer))
    }
  )
)


## the list of all the target metanames ----

c(
  targets_raw_data,
  targets_cleaned_data
)
