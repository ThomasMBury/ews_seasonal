#!/bin/bash

mkdir -p output
mkdir -p output/stdout

for i in 0.001 0.002 0.003 0.004 0.005; do
    sbatch single_job_hag.sh $i
    sleep 1.0
done

