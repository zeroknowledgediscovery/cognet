from mpi4py.futures import MPIPoolExecutor
import numpy as np
import pandas as pd
from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix
from quasinet.qsampling import qsample, targeted_qsample
import sys

yr=sys._xoptions['y']
Nstr=sys._xoptions['N']
NAME_PREF='../ideology_2/ALL_Qdist_Matrices/ALL_DMAT_test'
if Nstr != 'all':
    N=int(Nstr)
else:
    N=None

GSS_dir = '../data/processed_data/'
POLEFILE='../polar_vectors.csv'
GSS_FEATURES_BY_YEAR='../data/features/features_by_year_GSS.csv'
GSSDATA=GSS_dir+'gss_'+yr+'.csv'
QPATH='../Qnets/gss_'+yr+'.joblib'

#qnet
qnet=load_qnet(QPATH)

#poles
poles=pd.read_csv(POLEFILE,index_col=0)
L=poles.L.to_dict()
R=poles.R.to_dict()
poles=poles.transpose()

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