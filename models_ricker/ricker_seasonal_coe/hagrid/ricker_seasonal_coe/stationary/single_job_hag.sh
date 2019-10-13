#!/bin/bash

# # Thomas Bury
# # PhD Candidate
# # Bauch computational epidemiology research group
# # Department of Applied Mathematics
# # Faculty of Mathematics
# # University of Waterloo


#SBATCH --mem=1000MB
#SBATCH --account=hagrid
#SBATCH --partition=hagrid_long
#SBATCH --time=0-01:00:00
#SBATCH --error="Jobs/job-%j.err"
#SBATCH --output=Jobs/job-%j.out
#SBATCH --ntasks=1

echo Job $SLURM_JOB_ID released

echo Install python modules
python3 -m pip install -r requirements.txt


mkdir -p Jobs/job-$SLURM_JOB_ID
cd Jobs/job-$SLURM_JOB_ID

time python3 ../../equi_search.py

cd ../../

