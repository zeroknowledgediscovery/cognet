#!/bin/bash
YEARS='2018'
# nodes requested
NODES=2
# time requested
T=14
NUM='all'
LAUNCH=./'mpi_launcher.sh'

for yr in `echo $YEARS`
do
	echo $yr
	./GSS_mpi_setup.sh $yr $NODES $NUM tmp_"$yr"
	$LAUNCH -P tmp_"$yr" -F -T $T -N "$NODES" -C 28 -p broadwl -J MPI_TMP_"$yr" -M 56
done
rm tmp_"$yr"*
