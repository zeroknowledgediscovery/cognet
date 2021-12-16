# testing cognet vs native Qnet timess
import sys

from quasinet.qnet import qdistance
from cognet.cognet import cognet as cg
from cognet.dataFormatter import dataFormatter
from cognet.model import model 
#import cognet.util
import pandas as pd
import numpy as np

from quasinet.qnet import Qnet
import multiprocessing
import time

yr = '2018'
POLEFILE='examples_data/polar_vectors.csv'
QPATH='examples_data/gss_'+yr+'.joblib'
IMMUTABLE_FILE='examples_data/immutable.csv'
GSSDATA = 'examples_data/gss_'+yr+'.csv'

# testing dataFormatter
data = dataFormatter(samples=GSSDATA)
# load the sample data
# have option for test/train split
# make checks to ensure we will not throw errors at qnet construction 
print(data.samples[:2])
features,samples = data.format_samples('train')
use_all_samples = True
if use_all_samples:
    features,samples = data.Qnet_formatter()
 # default trains and tests using half
print(samples.shape)

# format data for Qnet training and fitting
data.Qnet_formatter()

# set mutable and immutable vars either from list or file
im_vars_df = pd.read_csv(IMMUTABLE_FILE, names=['vars'])
im_vars_list = im_vars_df.vars.to_list()
mutable_vars, immutable_vars = data.mutable_variables(immutable_list=im_vars_list)
mutable_vars, immutable_vars = data.mutable_variables(IMMUTABLE_FILE=IMMUTABLE_FILE)

# testing model functionality
# can either input features and samples directly, or infer from data obj
model_ = model()
# qnet construction parameters, 
# choose to either load or fit qnet from scratch
test_model_buildqnet = True
if test_model_buildqnet:
        start = time.time()
        print("fitting")
        model_.fit(data_obj=data,
                   njobs=6)
        end = time.time()
        print("cognet time: ", end-start)
        print("fitted")
        model_.export_dot("examples_results/tmp_dot_modelclass.dot",
                        generate_trees=True)
        model_.save("examples_data/tmp_nodelclass.joblib")
        #model_.load("tmp_nodelclass.joblib")
else:
    model_.load("examples_data/gss_2018.joblib")

features,samples = data.format_samples('train')
test_samples = samples[:2]

print(multiprocessing.cpu_count())
start = time.time()
Qnet_ = Qnet(n_jobs=6, feature_names=features)
print("fitting")
start = time.time()
print("samples: ", samples)
Qnet_.fit(samples)
end = time.time()
print("Native Qnet fitting time: ", end-start)
