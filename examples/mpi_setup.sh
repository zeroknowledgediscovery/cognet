#!/bin/bash
YEAR=$1
	NODES=$2
else
	NODES=3
fi
if [ $# -gt 2 ] ; then
	NUM=$3
else
	NUM='all'
fi
if [ $# -gt 3 ] ; then
	PROG=$4
else
	PROG=$(tty)


fi
NUMPROC=`expr 28 \* $NODES`
echo "module load midway2" >> $PROG
echo "module unload python" >> $PROG
echo "module unload openmpi" >> $PROG
echo "module load python/anaconda-2020.02" >> $PROG
echo "module load mpi4py" >> $PROG
echo "date; mpiexec -n "$NUMPROC" python3 -Xy="$YEAR" -XN="$NUM" -m mpi4py.futures tmp_dmat.py; date"  >> $PROG
