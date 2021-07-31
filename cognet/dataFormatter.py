# train subset of samples
import numpy as np
import pandas as pd
import random
from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix
from quasinet.qsampling import qsample, targeted_qsample
import os
os.system("module unload openmpi")
#from mpi4py.futures import MPIPoolExecutor
import sys
import subprocess
from pqdm.processes import pqdm
from scipy.stats import entropy

import multiprocessing as mp
import time
from sklearn.model_selection import train_test_split
class dataFormatter:
    """Aggregate related Qnet functions
    """

    def __init__(self,
                 samples,
                 test_size,
                 train_size=None,
                 random_state=None):
        """[summary]

        Args:
            samples ([type]): [description]
            test_size ([type]): [description]
            train_size ([type]): [description]
            random_state ([type], optional): [description]. Defaults to None.
        """
        self.samples = samples
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = None
        self.train_data, self.test_data = train_test_split(samples,
                                                           test_size=test_size,
                                                           train_size=train_size,
                                                           random_state=random_state)
        self.features = {}
        self.nan_cols = []

    def __Qnet_formatter(self,
                         key,
                         samples):
        """[summary]

        Args:
            key ([type]): [description]
            samples ([type]): [description]

        Returns:
            [type]: [description]
        """
        features = np.array(self.samples.columns)
        samples = samples.values.astype(str)[:]
        # remove columns that are all NaNs
        not_all_nan_cols = ~np.all(X == '', axis=0)
        self.nan_cols = np.all(X == '', axis=0)

        samples = samples[:, not_all_nan_cols]
        
        features = features[not_all_nan_cols]
        features = list(features)
        self.features[key] = features
        return features, samples

    def train(self):
        """[summary]
        """
        return self.__Qnet_formatter('train',self.train_data)
    
    def test(self):
        """[summary]
        """
        return self.__Qnet_formatter('test',self.test_data)
    
    def set_immutable_vars(self,
                           immutable_list=None,
                           IMMUTABLE_FILE=None,
                           mutable_list=None,
                           MUTABLE_FILE=None):
        '''
        set vars to immutable and mutable, 
        can prob combine this with the load_data func: only set the immutable vars if necessary

        args:
            IMMUTABLE_FILE (str): file containing the immutable features/vars
        '''
        list_None = assert_None([immutable_list,mutable_list], raise_error=False)
        file_None = assert_None([IMMUTABLE_FILE,MUTABLE_FILE], raise_error=False)
        num_None = assert_None([immutable_list,mutable_list,
                                IMMUTABLE_FILE,MUTABLE_FILE], raise_error=False)
        if list_None == 2 or file_None == 2:
            raise ValueError("Only input either IMMUTABLE or MUTABLE vars, not both!")
        elif num_None != 1:
            raise ValueError("Too many inputs! Only one argument needed")
        else:
            

        if self.cols is None:
            raise ValueError("load_data first!")
        self.immutable_vars = pd.read_csv(IMMUTABLE_FILE,index_col=0).transpose()
        self.mutable_vars = None
        self.mutable_vars = [x for x in self.cols
                             if x.upper() not in self.immutable_vars.columns]