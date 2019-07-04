#!/bin/bash

rm par_table.txt
touch par_table.txt

declare -a TMAX_VALS=(200 500 1000);
declare -a RW_VALS=(0.2 0.4);
declare -a DT2_VALS=(1 2);


echo "tmax rw dt2" >> par_table.txt;

for tmax in "${TMAX_VALS[@]}"; do
	for rw in "${RW_VALS[@]}"; do
		for dt2 in "${DT2_VALS[@]}"; do
			echo "$tmax $rw $dt2" >> par_table.txt;
		done
	done
done

