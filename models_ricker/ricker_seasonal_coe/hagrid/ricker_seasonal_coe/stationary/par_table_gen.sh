#!/bin/bash

rm par_table.txt
touch par_table.txt

declare -a SIGMA_VALS=(0.05);
declare -a A_VALS=(0.0005 0.001 0.002 0.004)


echo "sigma a" >> par_table.txt;

for sigma in "${SIGMA_VALS[@]}"; do
	for a in "${A_VALS[@]}"; do
		echo "$sigma $a" >> par_table.txt;
	done
done

