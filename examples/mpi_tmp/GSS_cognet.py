from mpi4py.futures import MPIPoolExecutor
import numpy as np
import pandas as pd
from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix
from quasinet.qsampling import qsample, targeted_qsample

qnet=load_qnet('examples_data/gss_2018.joblib')
w = 10
h = w
p_all = pd.read_csv("tmp_samples_as_strings.csv", header=None)

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