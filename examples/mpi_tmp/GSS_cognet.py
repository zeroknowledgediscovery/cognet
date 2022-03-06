from mpi4py.futures import MPIPoolExecutor
import numpy as np
import pandas as pd
from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix
from quasinet.qsampling import qsample, targeted_qsample

qnet=load_qnet('../GSS/data/gss_2018.joblib')
w = 1784
h = w
p_all = pd.read_csv("tmp_samples_as_strings.csv", header=None)

POLEFILE='../GSS/data/polar_vectors.csv'
GSS_FEATURES_BY_YEAR='../GSS/data/features_by_year_GSS.csv'
yr = '2018'
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
print("________________________________________________________")
p_all.columns = cols
for x in p_all.columns:
    if x not in poles.columns:
        p_all[x] = ""
# p_all=p_all[[x for x in poles.columns if x in p_all.columns]]
# for x in poles.columns:
#     if x not in p_all.columns:
#         p_all[x]=np.nan

features = pd.DataFrame(columns=cols)
all_features = pd.concat([poles, features], axis=0)
pL= all_features.loc['L'][cols].fillna('').values.astype(str)[:]
pR= all_features.loc['R'][cols].fillna('').values.astype(str)[:]
p_all=pd.DataFrame(p_all.fillna('').values.astype(str)[:])

def distfunc(x,y):
	d=qdistance(x,y,qnet,qnet)
	return d

def dfunc_line(k):
	line = np.zeros(w)
	y = np.array(p_all.iloc[k])
	for j in range(w):
		if j > k:
			x = np.array(p_all.iloc[j])
			line[j] = distfunc(x, y)
	return line

if __name__ == '__main__':
	with MPIPoolExecutor() as executor:
		result = executor.map(dfunc_line, range(h))
		pd.DataFrame(result).to_csv('GSS_2018_distmatrix.csv',index=None,header=None)