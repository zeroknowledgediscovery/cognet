# This is an example of how we see 
# the package work. The functions listed here
# are probably the only ones that should be exposed, ie documented.
# others should br prepended with a double underscore
#  
# The cognet directory has the following "modules"
# which are seprate .py files containing clases and functions
# The modules are cognet.py, dataFormatter.py, model.py, util.py, viz.py
# we will write the viz.py later.
import sys

from quasinet.qnet import qdistance
from cognet.cognet import cognet as cg
from cognet.dataFormatter import dataFormatter
from cognet.model import model 
#import cognet.util
import pandas as pd
import numpy as np

yr = '2018'
POLEFILE='../GSS/data/polar_vectors.csv'
QPATH='../GSS/data/gss_'+yr+'.joblib'
IMMUTABLE_FILE='../GSS/data/immutable.csv'
GSSDATA = '../GSS/data/gss_'+yr+'.csv'

# testing dataFormatter
data = dataFormatter(samples=GSSDATA)
# load the sample data
# have option for test/train split
# make checks to ensure we will not throw errors at qnet construction 
print(data.samples[:2])
features,samples = data.format_samples('train') # default trains and tests using half
all_samples = True
if all_samples: # use all samples to train, instead of half
    features,samples = data.Qnet_formatter()

model_ = model()
QNETFILE = '../GSS/data/gss_2018.joblib'
model_.load(QNETFILE)

Cg = cg()
Cg.load_from_model(model_, data, 'all')
## embedding
## embed generated Qdist Matrix
Cg.year = '2018'
Cg.embed('GSS_2018_distmatrix.csv', 'embed', '',EMBED_BINARY='../../../cognet/cognet/bin/__embed__.so')
#pd.read_csv('examples_results/embed_E_2018.csv')
