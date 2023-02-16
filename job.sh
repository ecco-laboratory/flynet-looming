#!/bin/bash

# Submit the pipeline as a job with srun job.sh

# Modified from https://github.com/mschubert/clustermq/blob/master/inst/LSF.tmpl
# under the Apache 2.0 license:
#SBATCH --account=default
#SBATCH --partition=day-long
#SBATCH --nodelist=node1
#SBATCH --time=0-23:59:59
#SBATCH --job-name=mthieu_r_targets
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem-per-cpu=4096
#SBATCH --cpus-per-task=1

module load R/4.2.2 # Comment out if R is not an environment module.
R CMD BATCH run.R

# Removing .RData is recommended.
rm -f .RData
