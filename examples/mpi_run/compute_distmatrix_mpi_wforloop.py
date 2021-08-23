# This is an example of how we see 
# the package work. The functions listed here
# are probably teh only ones that should be exposed, ie documented.
# others should br prepended with a double underscore
#  
# The cognet directory has the following "modules"
# which are seprate .py files containing clases and functions
from mpi4py.futures import MPIPoolExecutor
import sys
sys.path.append("../cognet")

from quasinet.qnet import qdistance
from cognet import cognet

from dataFormatter import dataFormatter
from model import model
import pandas as pd
import numpy as np

yr = '2018'
POLEFILE='../examples_data/polar_vectors.csv'
QPATH='../examples_data/gss_'+yr+'.joblib'
IMMUTABLE_FILE='../examples_data/immutable.csv'
GSSDATA = '../examples_data/gss_'+yr+'.csv'

commandline_test = True
#train data object
data = dataFormatter(samples=GSSDATA,
                test_size=0.5)
features,samples = data.train()

im_vars_df = pd.read_csv(IMMUTABLE_FILE, names=['vars'])
im_vars_list = im_vars_df.vars.to_list()
mutable_vars, immutable_vars = data.mutable_variables(immutable_list=im_vars_list)
mutable_vars, immutable_vars = data.mutable_variables(IMMUTABLE_FILE=IMMUTABLE_FILE)

model_ = model()
model_.load("../examples_data/gss_2018.joblib")

Cg = cognet()
Cg.load_from_model(model_, samples_file='../examples_data/gss_2018.csv')

# produce stats on how many column names actually match
stats = Cg.set_poles(POLEFILE,steps=2)
Cg.num_qsamples = 5

if commandline_test:
    print("testing mpiexex print error")
    Cg.set_nsamples(10)
#distance_matrix=Cg.distfunc_multiples("distfunc_multiples_testing.csv")

return_dict={}
for i in range(len(Cg.samples)):
    return_dict[i]=Cg.distfunc_line(i)
result=[x for x in return_dict.values()]
columns = [i for i in range(len(Cg.samples))]
result=pd.DataFrame(result,columns=columns, index=columns).sort_index(ascending=False)
# result = result.to_numpy()
# result = pd.DataFrame(np.maximum(result, result.transpose()))
result.to_csv('../examples_results/distfunc_forloop_compute_node.csv',index=None,header=None)