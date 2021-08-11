#!/bin/bash
# YEARS='1986 1978 2018 2010 2014 1987 2012 1985 2006 2016 1989 1982 2004 2008 1980 1996 1975 1990 1994 1973 2002 1972 1974 1977 1991 1998 1983 1993 2000 1984 1976 1988'
YEARS='2018'
# nodes requested
NODES=200
# number of samples (u can set it toa number like 10)
NUM='all'
if [ $# -gt 0 ] ; then
    NUM=$1
fi
# time requested
T=127


LAUNCH='../../../LAUNCH_UTILITY/launcher_s.sh'
for yr in `echo $YEARS`
do
    echo $yr
    ./mpi_dcal_setup.sh $yr $NODES $NUM tmp_"$yr"
    $LAUNCH -P tmp_"$yr" -F -T $T -N "$NODES" -C 28 -p broadwl -J XXXXXXXtmpACRD_"$yr" -M 64 -d 1 

done
rm tmp*

