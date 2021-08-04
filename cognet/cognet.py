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

class cognet:
    """Aggregate related Qnet functions
    """

    def __init__(self):
        """Init
        """
        self.year = None
        self.n_jobs = 28
        self.qnet = None
        self.steps = 120
        self.num_qsamples = None
        self.samples = None
        self.samples_as_strings = None
        self.features = None
        self.cols = None
        self.immutable_vars = None
        self.mutable_vars = None
        self.poles = None
        self.polar_features = None
        self.polar_indices = None
        self.pL = None
        self.pR = None
        self.d0 = None
        self.qdistance_matrix_file = None
        self.dissonance_file = None
        self.s_null = None
        self.D_null = None
        self.mask_prob = 0.5
        self.variation_weight = None
        self.polar_matrix = None
    
    def load_from_model(self,
                        model,
                        im_vars=None,
                        m_vars=None):
        """[summary]

        Args:
            model ([type]): [description]
            im_vars ([type], optional): [description]. Defaults to None.
            m_vars ([type], optional): [description]. Defaults to None.
        """
        elif model is not None:
            self.qnet = model.myQnet
            self.cols = model.features
            self.features = pd.DataFrame(columns=self.cols)
            self.immutable_vars = model.immutable_vars
            self.mutable_vars = model.mutable_vars
            
            samples = pd.DataFrame(model.samples)
            self.samples = pd.concat([samples,self.features], axis=0)
            self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
            self.s_null=['']*len(self.samples_as_strings[0])
            self.D_null=self.qnet.predict_distributions(self.s_null)
            variation_weight = []
            for d in self.D_null
                v=[]
                for val in d_.values():
                    v=np.append(v,val)
                variation_weight.append(entropy(v,base=len(v)))
            self.variation_weight = variation_weight
            
    def load_data(self,
                  year,
                  features_by_year,
                  samples,
                  qnet):
        '''load cols, features, samples, and qnet.

        Args:
          year (str): to identify cols/features.
          features_by_year (str): file containing all features by year of the dataset.
          samples (str): file of samples for that year.
          Qnet (str): Qnet file location.
        '''
        self.qnet = load_qnet(qnet)
        self.year = year
        self.cols = np.array((pd.read_csv(features_by_year,
                               keep_default_na=True, 
                               index_col=0).set_index(
                                   'year')).loc[int(year)].apply(
                                       eval).values[0])
        self.features = pd.DataFrame(columns=self.cols)
        self.mutable_vars = [x for x in self.cols]
        #[self.cols].fillna('').values.astype(str)[:]

        self.samples=pd.read_csv(samples)
        self.samples = pd.concat([self.samples,self.features], axis=0)
        self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
        self.s_null=['']*len(self.samples_as_strings[0])
        self.D_null=self.qnet.predict_distributions(self.s_null)
        variation_weight = []
        for d in self.D_null:
            v=[]
            for val in d.values():
                v=np.append(v,val)
            variation_weight.append(entropy(v,base=len(v)))
        self.variation_weight = variation_weight

    def set_immutable_vars(self,
                           IMMUTABLE_FILE):
        '''
        set vars to immutable and mutable, 
        can prob combine this with the load_data func: only set the immutable vars if necessary

        Args:
          IMMUTABLE_FILE (str): file containing the immutable features/vars
        '''
        if self.cols is None:
            raise ValueError("load_data first!")
        self.immutable_vars = pd.read_csv(IMMUTABLE_FILE,index_col=0).transpose()
        self.mutable_vars = None
        self.mutable_vars = [x for x in self.cols
                             if x.upper() not in self.immutable_vars.columns]
    
    def set_nsamples(self,
                     num_samples):
        '''
        select a subset of the samples

        Args:
          num_samples (int): Set num of samples to subset
        '''
        
        if all(x is not None for x in [num_samples, self.samples]):
            if num_samples > len(self.samples.index):
                string = 'The number of selected samples ({}) ' + \
                     'is greater than the number of samples ({})!'
                string = string.format(num_samples, len(self.samples.index))
                raise ValueError(string)

            if num_samples == len(self.samples.index):
                string = 'The number of selected samples ({}) ' + \
                     'is equal the number of samples ({})!'
                string = string.format(num_samples, len(self.samples.index))
                print(string)
            #self.samples = self.samples.sample(num_samples)
            self.samples = self.samples[:10]
            self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]

        elif self.samples is None:
            raise ValueError("load_data first!")

    def __variation_weight(self,
                           index):
        """[summary]

        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        d_=self.D_null[index]
        v=[]
        for val in d_.values():
            v=np.append(v,val)
        return entropy(v,base=len(v))
    
    def getBaseFrequency(self, 
                         sample):
        '''
        get frequency of the variables
        helper func for qsampling

        Args:
          sample (list[str]): vector of sample, must have the same dimensions as the qnet
        '''
        MUTABLE=pd.DataFrame(np.zeros(len(self.cols)),index=self.cols).transpose()
                
        for m in self.mutable_vars:
            MUTABLE[m]=1.0
        mutable_x=MUTABLE.values[0]
        base_frequency=mutable_x/mutable_x.sum()

        # for i in range(len(base_frequency)):
        #     if base_frequency[i]>0.0:
        #         base_frequency[i]= self.__variation_weight(i)*base_frequency[i]

        return base_frequency/base_frequency.sum()

    def qsampling(self,
                  sample,
                  steps,
                  immutable=False):
        '''
        perturb the sample based on thet qnet distributions and number of steps

        Args:
          sample (1d array-like): vector of sample, must have the same dimensions as the qnet
          steps (int): number of steps to qsample
          immutable (bool): are there variables that are immutable?
        '''
        if all(x is not None for x in [self.mutable_vars, self.immutable_vars, sample]):
            if immutable == True:
                return qsample(sample,self.qnet,steps,self.getBaseFrequency(self.samples))
            else:
                return qsample(sample,self.qnet,steps)
        elif self.mutable_vars is None:
            raise ValueError("load_data first!")
        elif self.immutable_vars is None:
            raise ValueError("load immutable variables first!")

    def set_poles(self,
                  POLEFILE,
                  steps=0,
                  mutable=False):
        '''
        set the poles and samples such that the samples contain features in poles


        Args:
          steps (int): number of steps to qsample
          POLEFILE (str): file containing poles samples and features
          mutable (boolean): Whether or not to set poles as the only mutable_vars
        '''
        invalid_count = 0
        if all(x is not None for x in [self.samples]):
            poles = pd.read_csv(POLEFILE, index_col=0)
            L=poles.L.to_dict()
            R=poles.R.to_dict()
            self.poles=poles.transpose()

            cols = [x for x in self.poles.columns if x in self.samples.columns]
            self.samples=self.samples[cols]
        
            for x in self.poles.columns:
                if x not in self.samples.columns:
                    invalid_count += 1
                    self.samples[x]=np.nan

            self.samples = pd.concat([self.samples,self.features], axis=0)
            self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]

            self.polar_features = pd.concat([self.poles, self.features], axis=0)
            pL= self.polar_features.loc['L'][self.cols].fillna('').values.astype(str)[:]
            pR= self.polar_features.loc['R'][self.cols].fillna('').values.astype(str)[:]

            self.pL=self.qsampling(pL,steps)
            self.pR=self.qsampling(pR,steps)
            if mutable:
                self.mutable_vars=[x for x in self.cols if x in self.poles.columns]
        elif self.samples is None:
            raise ValueError("load_data first!")
        
        print("{} pole features not found in sample features".format(invalid_count))

    def distance(self,
                 sample1,
                 sample2,
                 nsteps1=0,
                 nsteps2=0):
        """[summary]

        Args:
            sample1 ([type]): [description]
            sample2 ([type]): [description]
            nsteps1 (int, optional): [description]. Defaults to 0.
            nsteps2 (int, optional): [description]. Defaults to 0.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if self.qnet is None:
            raise ValueError("load qnet first!")
        sample1 = pd.DataFrame(sample1).fillna('').values.astype(str)[:]
        sample2 = pd.DataFrame(sample1).fillna('').values.astype(str)[:]
        bp1 = self.getBaseFrequency(sample1)
        bp2 = self.getBaseFrequency(sample2)
        sample1 = qsample(sample1, self.qnet, nsteps1, baseline_prob=bp1)
        sample2 = qsample(sample2, self.qnet, nsteps2, baseline_prob=bp2)
        return qdistance(sample1, sample2)
    
    def __distfunc(self, 
                 x, 
                 y):
        '''
        Compute distance between two samples

        Args:
          x : first sample
          y : second sample
        '''
        d=qdistance(x,y,self.qnet,self.qnet)
        return d
    
    def polarDistance(self,
                      sample):
        """[summary]

        Args:
            sample ([type]): [description]

        Returns:
            [type]: [description]
        """
        distances = {}
        for index, row in self.poles.iterrows():
            distances[index] = distance(sample, row)
        return distances
            
    
    def distfunc_line(self,
                   row):
        '''
        compute the dist for a row, or vector of samples

        Args:
          k (int): row
        return:
          numpy.ndarray(float)
        '''
        if all(x is not None for x in [self.samples, self.features]):
            w = self.samples.index.size
            p_all = pd.concat([self.samples, self.features], axis=0)[cols].fillna('').values.astype(str)[:]
            line = np.zeros(w)
            y = p_all[row]
            for j in range(w):
                # only compute half of the distance matrix
                if j > row:
                    x = p_all[j]
                    line[j] = self.__distfunc(x, y)
        else:
            raise ValueError("load_data first!")
        return line
    
    def polar_separation(self,
                         nsteps=0):
        """[summary]

        Args:
            nsteps (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        polar_arraydata = self.polar_features[self.cols].fillna('').values.astype(str)[:]
        print("testing polar_separation")
        print(polar_arraydata)
        samples_ = []
        for vector in polar_arraydata:
            bp = self.getBaseFrequency(vector)
            sample = qsample(vector, self.qnet, nsteps, baseline_prob=bp)
            samples_.append(sample)
        self.polar_matrix = qdistance_matrix(samples_, samples_, self.qnet, self.qnet)
        return self.polar_matrix
        
    def embed(self,
              infile,
              name_pref,
              out_dir):
        '''
        embed data

        Args:
          infile (str): input file to be embedded
          name_pref (str): preferred name for output file
          out_dir (str): output dir for results
        '''
        if all(x is not None for x in [self.year]):
            yr = self.year
            PREF = name_pref
            FILE = infile

            EMBED = '../GSS/bin/embed'
            DATAFILE = 'data_' +yr
            EFILE = out_dir + PREF + '_E_' +yr
            DFILE = out_dir + PREF + '_D_' +yr

            pd.read_csv(infile,header=None).to_csv(out_dir + DATAFILE,sep=' ',header=None,index=None)
            STR=EMBED+' -f '+DATAFILE+' -E '+EFILE+' -D '+DFILE
            subprocess.call(STR,shell=True)
        elif self.year is None:
            raise ValueError("load_data first!")
    
    def compute_DLI_sample(self,
                           i,):
        '''
        return ideology index, dL, dR, Qsd (std), Q (max) for one sample

        Args:
          i (int): index of sample
        '''
        p = self.samples_as_strings[i]
        dR = qdistance(self.pR, p, self.qnet, self.qnet)
        dL = qdistance(self.pL, p, self.qnet, self.qnet)
        ideology_index = (dR-dL)/self.d0
        
        Qset = [qsample(p, self.qnet, self.steps) for j in np.arange(self.num_qsamples)]
        Qset = np.array(Qset)

        matrix = (qdistance_matrix(Qset, Qset, self.qnet, self.qnet))
        Q = matrix.max()
        Qsd = matrix.std()

        return [ideology_index, dL, dR, Qsd, Q]
       
    def compute_DLI_samples(self,
                    num_qsamples,
                    outfile,
                    steps=5,
                    n_jobs=28):
        '''
        compute and save ideology index, dL, dR, Qsd (std), Q (max) for all samples

        Args:
          num_qsamples (int): number of qsamples to compute
          steps (int): number of steps to qsample
          outfile (str): output file for results
        '''
        if all(x is not None for x in [self.samples, self.features,
                                    self.pL, self.pR]):
            self.num_qsamples = num_qsamples
            self.steps = steps
            self.d0 = qdistance(self.pL, self.pR, self.qnet, self.qnet)

            result=pqdm(range(len(self.samples)), self.compute_DLI, n_jobs)
            pd.DataFrame(result,
                        columns=['ido', 'dL', 'dR', 'Qsd', 'Q']).to_csv(outfile)

        elif self.pL is None or self.pR is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")
    
    def compute_polar_indices(self,
                              num_samples = None,
                              polar_comp = False,
                              POLEFILE = None,
                              steps = 5):
        '''
        set up polar indices for dissonance func

        Args:
          num_samples (int): subset of samples to take
          polar_comp (bool): whether or not to set poles
          POLEFILE (None): file containing pole samples and features
          steps (int): number of steps to qsample
        '''
        if all(x is not None for x in [self.samples, self.features, self.poles]):
            if num_samples is not None:
                self.set_nsamples(num_samples)

            # read sample data
            if polar_comp:
                self.set_poles(self.qnet, steps, POLEFILE)
            
            polar_features = pd.concat([self.poles, self.features], axis=0)
            self.polar_indices=np.where(polar_features[self.cols].fillna('XXXX').values[0]!='XXXX')[0]
        
        elif self.poles is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")

    def dissonance(self,
                sample_index,
                MISSING_VAL=0.0):
        '''
        compute dissonance for each sample_index, helper function for all_dissonance
        
        Args:
          sample_index (int): index of the sample to compute dissonance
          MISSING_VAL (float): 
        '''
        if all(x is not None for x in [self.samples, self.features, 
                                       self.poles]):
            s = self.samples_as_strings[sample_index]
            if self.polar_indices is None:
                self.polar_indices = range(len(s))

            Ds=self.qnet.predict_distributions(s)
            
            diss=np.ones(len(Ds))*MISSING_VAL
            for i in self.polar_indices:
                if s[i] != '':
                    if s[i] in Ds[i].keys():
                        diss[i]=1-Ds[i][s[i]]/np.max(
                            list(Ds[i].values())) 
                    else:
                        diss[i]=1.0
            return diss[self.polar_indices]

        elif self.poles is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")
    
    def dissonance_matrix(self,
                          output_file='DISSONANCE_'+self.year+'.csv',
                          n_jobs=28):
        '''
        get the dissonance for all samples

        Args:
          output_file (str): directory and/or file for output
          n_jobs (int): number of jobs for pdqm
        '''
        result=pqdm(range(len(self.samples)), self.dissonance, n_jobs)
        out_file = output_file

        pd.DataFrame(result,
                    columns=self.polar_features[self.cols].dropna(
                    axis=1).columns).to_csv(out_file)
        self.dissonance_file = out_file
    
    def __choose_one(self,
                   X):
        '''
        returns a random element of X

        Args:
          X (1D array-like): vector from which random element is to be chosen
        '''
        X=list(X)
        if len(X)>0:
            return X[np.random.randint(len(X))]
        return None

    def getMaskedSample(self,
                        s,
                        mask_prob=0.5,
                        allow_all_mutable=False):
        '''
        inputs a sample and randomly mask elements of the sample

        Args:
          s (list[str]): vector of sample, must have the same dimensions as the qnet
          mask_prob (float): float btwn 0 and 1, prob to mask element of sample
          allow_all_mutable (bool): whether or not all variables are mutable
        '''
        if self.samples is not None:   
            MUTABLE=pd.DataFrame(np.zeros(len(self.cols)),index=self.cols).transpose()
            WITHVAL=[x for x in self.cols[np.where(s)[0]] if x in self.mutable_vars ]
            MASKrand=[x for x in WITHVAL if random.random() < mask_prob ]
            for m in MASKrand:
                MUTABLE[m]=1.0
            
            mutable_x=MUTABLE.values[0]
            base_frequency=mutable_x/mutable_x.sum()

            # if np.isnan(base_frequency).any():
            #     return np.nan,np.nan,np.nan
            #     return self.getMaskedSample(s)

            s1=s.copy()
            for i in range(len(base_frequency)):
                if base_frequency[i]>0.0001:
                    s1[i]=''
                    
            s_rand=np.copy(s)
            rnd_match_prob=[]        
            max_match_prob=[]        
            D=self.qnet.predict_distributions(s)
            for i in MASKrand:
                s_rand[np.where(
                    self.cols==i)[0][0]]=self.__choose_one(
                        self.D_null[np.where(self.cols==i)[0][0]].keys())
                rnd_match_prob=np.append(rnd_match_prob,1/len(
                    self.D_null[np.where(self.cols==i)[0][0]].keys()))
                max_match_prob=np.append(
                    max_match_prob,np.max(
                        list(D[np.where(
                            self.cols==i)[0][0]].values())))
                
            if allow_all_mutable:
                for m in mutable_vars:
                    MUTABLE[m]=1.0
                mutable_x=MUTABLE.values[0]
                base_frequency=mutable_x/mutable_x.sum()

            return s1,base_frequency,MASKrand,np.where(
                base_frequency)[0],np.mean(rnd_match_prob),np.mean(max_match_prob),s_rand
        else:
            raise ValueError("load_data first!")

    def predict_maskedsample(self,
                             sample=None,
                             index=None,
                             return_dict):
        '''
        reconstruct the masked sample by qsampling and comparing to original
        set self.mask_prob and self.steps if needed

        Args:
          index (int): index of sample to take
        '''
        if all(x is None for x in [sample, index]):
            raise ValueError("Must input either sample or index!")
        elif all(x is not None for x in [sample, index]):
            raise ValueError("Must input either sample or index not both!")
        elif sample is not None:
            s=pd.DataFrame(sample).fillna('').values.astype(str)[:]
        elif index is not None:
            s=self.samples_as_strings[index]
            
        s1,bp,mask_,maskindex,rmatch_u,rmatch,s_rand=self.getMaskedSample(s, 
                                                                          mask_prob=self.mask_prob)
        if np.isnan(bp).any():
            return_dict[index] = np.nan,np.nan,np.nan
            return np.nan,np.nan,np.nan

        qs=qsample(s1,self.qnet,self.steps,bp)

        dqestim=qdistance(s,qs,self.qnet,self.qnet)
        dactual=qdistance(s,s1,self.qnet,self.qnet)
        qdistance_time_end = time.time()

        return_dict[index] = (1 - (dqestim/dactual))*100,rmatch_u,rmatch
        return (1 - (dqestim/dactual))*100,rmatch_u,rmatch

    def predict_maskedsamples(self):
        '''
        runs and saves the results of the predicted masked sample

        Args:
        '''
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for i in range(len(self.samples)):
            p = mp.Process(target=self.randomMaskReconstruction, args=(i, return_dict))
            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]

        result=[x for x in return_dict.values() if isinstance(x, tuple)]
        result=pd.DataFrame(result,columns=['rederr','r_prob','rand_err'])
        result.rederr=result.rederr.astype(float)

        if self.poles is not None:
            result.to_csv('Qnet_Constructor_tmp/rederror_first10_test'+self.year+str(self.steps)+'.csv')
        else:
            result.to_csv('Qnet_Constructor_tmp/polar_unrestrict_rederror'+self.year+str(self.steps)+'.csv')
        
        return result.rederr.mean(), result.rand_err.mean()