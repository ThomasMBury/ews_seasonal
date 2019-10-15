#!/bin/bash

mkdir -p output
mkdir -p output/stdout

. par_table_gen.sh

MAX=`cat par_table.txt | wc -l`

for i in `seq 2 $MAX`; do
	sbatch single_job_hag.sh $i
	sleep 1.0
	
done

