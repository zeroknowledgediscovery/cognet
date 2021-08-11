# This is an example of how we see 
# the package work. The functions listed here
# are probably teh only ones that should be exposed, ie documented.
# others should br prepended with a double underscore
#  
# The cognet directory has the following "modules"
# which are seprate .py files containing clases and functions
import sys
sys.path.append("../cognet")

from quasinet.qnet import qdistance
from cognet import cognet

from dataFormatter import dataFormatter
from model import model
import pandas as pd
import numpy as np

yr = '2018'
POLEFILE='examples_data/polar_vectors.csv'
QPATH='examples_data/gss_'+yr+'.joblib'
IMMUTABLE_FILE='examples_data/immutable.csv'
GSSDATA = 'examples_data/gss_'+yr+'.csv'

yr=sys._xoptions['y']
Nstr=sys._xoptions['N']
NAME_PREF='../ideology_2/ALL_Qdist_Matrices/ALL_DMAT_test'
if Nstr != 'all':
    N=int(Nstr)
else:
    N=None

#train data object
data = dataFormatter(samples=GSSDATA,
                test_size=0.5)
features,samples = data.train()

im_vars_df = pd.read_csv(IMMUTABLE_FILE, names=['vars'])
im_vars_list = im_vars_df.vars.to_list()
mutable_vars, immutable_vars = data.mutable_variables(immutable_list=im_vars_list)
mutable_vars, immutable_vars = data.mutable_variables(IMMUTABLE_FILE=IMMUTABLE_FILE)

model_ = model()
model_.load("examples_data/gss_2018.joblib")

Cg = cognet()
Cg.load_from_model(model_, samples_file='examples_data/gss_2018.csv')

# produce stats on how many column names actually match
stats = Cg.set_poles(POLEFILE,steps=2)

# compute polar distance matrix
dmatrix = Cg.polar_separation(nsteps=0)

# the following are for single samples
#------------------
dissonance_array = Cg.dissonance(1)
returndict = {}
rederr,r_prob,rand_err = Cg.randomMaskReconstruction(1, returndict)# sample=np.array(samples[1]))
#ideology_index = Cg.compute_DLI_sample(3)
Cg.num_qsamples = 5
ideology_index = Cg.ideology(3)

# get dispersion of an individual sample
Dispersion_ = Cg.dispersion(3)
# compute distance from each pole
array_distances = Cg.polarDistance(1, returndict)

#distance_matrix=Cg.distfunc_multiples("distfunc_multiples_testing.csv")

if __name__ == '__main__':

    with MPIPoolExecutor() as executor:
        result = executor.map(Cg.distfunc_line, range(len(Cg.samples)))
        pd.DataFrame(result).to_csv('distfunc_test.csv',index=None,header=None)