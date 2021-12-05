#!/bin/bash
YEARS='2018'
# nodes requested
NODES=2
# time requested
T=12
NUM='all'
LAUNCH='../../mpi_launcher.sh'

for yr in `echo $YEARS`
do
	echo $yr
	./cognet_example_setup.sh $yr $NODES $NUM tmp_"$yr"
	$LAUNCH -P tmp_"$yr" -F -T $T -N "$NODES" -C 28 -p broadwl -J QNET_TEST_6jobs_"$yr" -M 56
done
rm tmp_"$yr"*
