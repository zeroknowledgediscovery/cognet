#!/bin/bash
YEARS='2016'
# nodes requested
NODES=4
# time requested
T=12
LAUNCH='../../../../LAUNCH_UTILITY/launcher_s.sh'

for yr in `echo $YEARS`
do
	echo $yr
	./mpi_setup.sh $yr $NODES $NUM tmp_"$yr"
	$LAUNCH -P tmp_"$yr" -F -T $T -N "$NODES" -C 28 -p broadwl -J ../examples_results/ACRDALL_"$yr" -M 56
done
rm tmp*
