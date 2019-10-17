#!/bin/bash

rm par_table.txt
touch par_table.txt

declare -a SIGMA_VALS=(0.01 0.02);
declare -a A_VALS=(0 0.001 0.005);


echo "sigma a" >> par_table.txt;

for sigma in "${SIGMA_VALS[@]}"; do
	for a in "${A_VALS[@]}"; do
		echo "$sigma $a" >> par_table.txt;
	done
done

