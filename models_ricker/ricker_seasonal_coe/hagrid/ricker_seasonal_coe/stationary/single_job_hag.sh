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
#SBATCH --time=0-05:00:00
#SBATCH --error=output/stdout/job-%j.err
#SBATCH --output=output/stdout/job-%j.out
#SBATCH --ntasks=1

echo Job $SLURM_JOB_ID released

echo Install python modules
pip install ewstools
# python3 -m pip install -r requirements.txt

mkdir -p output/job-$1
cd output/job-$1

echo Run python file
python3 ../../sim_stat_hag.py $1

cd ../../

