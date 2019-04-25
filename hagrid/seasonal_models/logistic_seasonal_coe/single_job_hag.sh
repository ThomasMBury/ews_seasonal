#!/bin/bash

# # Thomas Bury
# # PhD Candidate
# # Bauch computational epidemiology research group
# # Department of Applied Mathematics
# # Faculty of Mathematics
# # University of Waterloo

#SBATCH --mem=1000MB
#SBATCH --time=1-00:00:00
#SBATCH --output=Jobs/output/job-%j.out
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tbury@uwaterloo.ca


echo Job $SLURM_JOB_ID released

# echo Install python modules
# python3 -m pip install -r requirements.txt

mkdir -p Jobs/job-$SLURM_JOB_ID
cd Jobs/job-$SLURM_JOB_ID

time python3 ../../script_compute_ews.py

cd ../../