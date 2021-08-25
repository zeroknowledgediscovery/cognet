
import sys
sys.path.append("../cognet")

from quasinet.qnet import qdistance
from cognet import cognet

from dataFormatter import dataFormatter
from model import model
import pandas as pd
import numpy as np
import dill as pickle

yr = '2016'
POLEFILE='examples_data/polar_vectors.csv'
QPATH='../../creed2_/GSS/Qnets/gss_'+yr+'.joblib'
IMMUTABLE_FILE='examples_data/immutable.csv'
GSSDATA = 'examples_data/processed_data/gss_'+yr+'.csv'

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
model_.load(QPATH)


#comm = MPI.COMM_WORLD
#rank = comm.rank
#if rank == 0:
#    Cg = cognet()
#    Cg.load_from_model(model_, samples_file='examples_data/gss_2018.csv')
#    stats = Cg.set_poles(POLEFILE,steps=2)
#    Cg.num_qsamples = 5
#    Cg.set_nsamples(10)
#else:
#    Cg = None
#
#Cg = comm.bcast(Cg, root=0)

Cg = cognet()
Cg.load_from_model(model_, samples_file=GSSDATA)
#
#### produce stats on how many column names actually match
stats = Cg.set_poles(POLEFILE,steps=2)
Cg.num_qsamples = 5
Cg.set_nsamples(500)

with open('examples_results/cgmod' + yr + '.mod', 'wb') as f:
    pickle.dump(Cg,f)
f.close()


