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

mkdir -p output/job-$SLURM_JOB_ID
cd output/job-$SLURM_JOB_ID

# Create text file with parameters specific to this job
head -n 1 ../../par_table.txt >> pars.txt
head -n $1 | tail -n 1 ../../par_table.txt >> pars.txt



echo Run python file
python3 ../../sim_stat_hag.py `head -n $1 | tail -n 1 ../../par_table.txt`

cd ../../

