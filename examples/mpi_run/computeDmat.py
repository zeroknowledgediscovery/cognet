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

# get features for each year
cols = (pd.read_csv(GSS_FEATURES_BY_YEAR,
                               keep_default_na=True, 
                               index_col=0).set_index(
                                   'year')).loc[int(yr)].apply(
                                       eval).values[0]
#read sample data
samples_=pd.read_csv(GSSDATA)
if N is not None:
    P=samples_.sample(N)
else:
    P=samples_
    
P_=P#[[x for x in poles.columns if x in P.columns]]
# for x in poles.columns:
#     if x not in P_.columns:
#         P_[x]=np.nan

#cols = cols_by_year[yr]
features = pd.DataFrame(columns=cols)
all_features = pd.concat([poles, features], axis=0)
pL= all_features.loc['L'][cols].fillna('').values.astype(str)[:]
pR= all_features.loc['R'][cols].fillna('').values.astype(str)[:]
p_all=pd.concat([P_, features], axis=0)[cols].fillna('').values.astype(str)[:]

w = P.index.size
h = w

def distfunc(x, y):
    d=qdistance(x,y,qnet,qnet)
    return d

def dfunc_line(k):
    '''
       args:
          k (int): row
      return:
          numpy.ndarray(float)
    '''
    line = np.zeros(w)
    y = p_all[k]
    for j in range(w):
        if j > k:
            x = p_all[j]
            line[j] = distfunc(x, y)
    return line

if __name__ == '__main__':

    with MPIPoolExecutor() as executor:
        result = executor.map(dfunc_line, range(h))
        pd.DataFrame(result).to_csv(NAME_PREF+yr+'.csv',index=None,header=None)