#!/bin/bash
YEAR=$1

if [ $# -gt 1 ] ; then
    NODES=$2
else
    NODES=2
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
echo "module load python/anaconda-2020.02" >> $PROG
echo "module unload python" >> $PROG
echo "module load python/cpython-3.7.0" >> $PROG
echo "module unload openmpi" >> $PROG
echo "module load mpi4py" >> $PROG
echo "date; mpiexec -n "$NUMPROC" python3 -Xy="$YEAR" -XN="$NUM" -m mpi4py.futures computeDmat.py; date"  >> $PROG
