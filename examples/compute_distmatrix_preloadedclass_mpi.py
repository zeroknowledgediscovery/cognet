# This is an example of how we see 
# the package work. The functions listed here
# are probably teh only ones that should be exposed, ie documented.
# others should br prepended with a double underscore
#  
# The cognet directory has the following "modules"
# which are seprate .py files containing clases and functions
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

import sys
sys.path.append("../cognet")

from quasinet.qnet import qdistance
from cognet import cognet

from dataFormatter import dataFormatter
from model import model
import pandas as pd
import numpy as np
import dill as pickle

yr = '2018'

with open('examples_results/cgmod.mod', 'rb') as f:
    Cg = pickle.load(f)


if __name__ == '__main__':

    with MPIPoolExecutor() as executor:
        result = executor.map(Cg.distfunc_line, range(len(Cg.samples)))
        print(result)
        pd.DataFrame(result).to_csv('examples_results/distfunc_mpitest_preloadedclass.csv',index=None,header=None)
