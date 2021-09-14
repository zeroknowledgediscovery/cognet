from mpi4py.futures import MPIPoolExecutor
import numpy as np
import pandas as pd
from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix
from quasinet.qsampling import qsample, targeted_qsample

qnet=load_qnet('examples_data/gss_2018.joblib')
w = 10
h = w
cols = (pd.read_csv('../../creed2_/GSS/data/features/features_by_year_GSS.csv',
                               keep_default_na=True, 
                               index_col=0).set_index(
                                   'year')).loc[int(2018)].apply(
                                       eval).values[0]
#read sample data
samples_=pd.read_csv('../../creed2_/GSS/data/processed_data/gss_2018.csv')
P=samples_.sample(10)
features = pd.DataFrame(columns=cols)
P_=P
p_all=pd.concat([P_, features], axis=0)[cols].fillna('').values.astype(str)[:]

def distfunc(x, y):
    print("distfunc running")
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
    print("line running")
    return line

print("running")
if __name__ == '__main__':

    with MPIPoolExecutor() as executor:
        result = executor.map(dfunc_line, range(h))
        pd.DataFrame(result).to_csv('tmp_distmatrix.csv',index=None,header=None)