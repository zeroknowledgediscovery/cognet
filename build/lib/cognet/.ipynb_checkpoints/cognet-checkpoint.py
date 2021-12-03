import numpy as np
import pandas as pd
import random
from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix
from quasinet.qsampling import qsample, targeted_qsample
#from mpi4py.futures import MPIPoolExecutor
import sys
import subprocess
from scipy.stats import entropy
import multiprocessing as mp
import time
from cognet.util import embed_to_pca
import pkgutil
import os

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
        self.all_samples = None
        self.samples = None
        self.samples_as_strings = None
        self.features = None
        self.cols = None
        self.immutable_vars = None
        self.mutable_vars = None
        self.poles = None
        self.polar_features = None
        self.polar_indices = None
        self.poles_dict = {}
        self.d0 = None
        self.qdistance_matrix_file = None
        self.dissonance_file = None
        self.s_null = None
        self.D_null = None
        self.mask_prob = 0.5
        self.variation_weight = None
        self.polar_matrix = None
        self.nsamples = None
        self.restricted = False
    
    def load_from_model(self,
                        model,
                        data_obj,
                        key,
                        im_vars=None,
                        m_vars=None):
        """load parameters from model object

        Args:
          model (Class): model obj for loading parameters
          data_obj (class): instance of dataformatter class
          key (str): 'all', 'train', or 'test', corresponding to sample type
          im_vars (list[str], optional): Not implemented yet. Defaults to None.
          m_vars (list[str], optional): Not implemented yet. Defaults to None.
        """
        if model is not None:
            self.qnet = model.myQnet
            # self.cols = np.array(model.features)
            featurenames, samples = data_obj.format_samples(key)
            samples = pd.DataFrame(samples)
            self.cols = np.array(featurenames)
            self.features = pd.DataFrame(columns=np.array(featurenames))
            if any(x is not None for x in [model.immutable_vars, model.mutable_vars]):
                if model.immutable_vars is not None:
                    self.immutable_vars = model.immutable_vars
                    self.mutable_vars = [x for x in self.features if x not in self.immutable_vars]
                elif model.mutable_vars is not None:
                    self.mutable_vars = model.mutable_vars
                    self.immutable_vars = [x for x in self.features if x not in self.mutable_vars]
            else:
                self.mutable_vars = self.features
            
            self.samples = pd.DataFrame(samples).replace("nan","").fillna("")
            self.samples.columns = np.array(featurenames)
            self.all_samples = self.samples
            self.samples_as_strings = self.samples.fillna('').values.astype(str)[:]
            self.s_null=['']*len(self.samples_as_strings[0])
            self.D_null=self.qnet.predict_distributions(self.s_null)
            variation_weight = []
            for d in self.D_null:
                v=[]
                for val in d.values():
                    v=np.append(v,val)
                variation_weight.append(entropy(v,base=len(v)))
            variation_weight = np.nan_to_num(variation_weight) # remove nans
            self.variation_weight = variation_weight
    
    def load_from_dataformatter(self, 
                                data_obj,
                                key):
        """read in either train or test data, specified by key, from data obj

        Args:
          data_obj (class): instance of dataformatter class
          key (str): 'all', 'train', or 'test', corresponding to sample type
          
        Returns:
          featurenames, samples: formatted arrays
        """
        featurenames, samples = data_obj.format_samples(key)
        if any(x is not None for x in [self.features, self.samples]):
            print("replacing original features/samples with dataformatter data")
        self.cols = featurenames
        self.features = pd.DataFrame(columns=self.cols)
        self.samples = pd.DataFrame(samples,columns=self.features)
        self.all_samples = self.samples
        self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
        self.s_null=['']*len(self.samples_as_strings[0])
        return featurenames, samples

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
        self.all_samples = self.samples
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
        '''set vars to immutable and mutable, 
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
                    num_samples,
                    random=False):
        '''select a subset of the samples

        Args:
          num_samples (int): Set num of samples to subset, default to None, resets to all samples
          random (bool): take random sample if true, ordered sample if false
        '''
        self.samples = self.all_samples
        if num_samples is not None:
            if all(x is not None for x in [num_samples, self.samples]):
                if num_samples > len(self.samples.index):
                    string = 'The number of selected samples ({}) ' + \
                        'is greater than the number of samples ({})!'
                    string = string.format(num_samples, len(self.samples.index))
                    raise ValueError(string)

                if num_samples == len(self.samples.index):
                    string = 'The number of selected samples ({}) ' + \
                        'is equal to the number of samples ({})!'
                    string = string.format(num_samples, len(self.samples.index))
                    print(string)
                if random:
                    self.samples = self.samples.sample(num_samples)
                else:
                    self.samples = self.samples.iloc[:num_samples]
                self.nsamples = num_samples
                self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
                
            elif self.samples is None:
                raise ValueError("load_data first!")

    def __variation_weight(self,
                        index):
        """
        """
        d_=self.D_null[index]
        v=[]
        for val in d_.values():
            v=np.append(v,val)
        return entropy(v,base=len(v))
    
    def getBaseFrequency(self, 
                        sample):
        '''get frequency of the variables
        helper func for qsampling

        Args:
          sample (list[str]): vector of sample, must have the same num of features as the qnet
        '''
        MUTABLE=pd.DataFrame(np.zeros(len(self.cols)),index=self.cols).transpose()
                
        for m in self.mutable_vars:
            MUTABLE[m]=1.0
        mutable_x=MUTABLE.values[0]
        base_frequency=mutable_x/mutable_x.sum()

        # commented out for now for testing using smaller qnet
        for i in range(len(base_frequency)):
            if base_frequency[i]>0.0:
                base_frequency[i]= self.variation_weight[i]*base_frequency[i]

        return base_frequency/base_frequency.sum()
    
    def qsampling(self,
                sample,
                steps,
                immutable=False):
        '''perturb the sample based on thet qnet distributions and number of steps

        Args:
          sample (1d array-like): sample vector, must have the same num of features as the qnet
          steps (int): number of steps to qsample
          immutable (bool): are there variables that are immutable?
        '''
        if all(x is not None for x in [self.mutable_vars, sample]):
            if immutable == True:
                return qsample(sample,self.qnet,steps,self.getBaseFrequency(self.samples))
            else:
                return qsample(sample,self.qnet,steps)
        elif self.mutable_vars is None:
            raise ValueError("load_data first!")

    def set_poles(self,
                  POLEFILE,
                  pole_1,
                  pole_2,
                  steps=0,
                  mutable=False,
                  VERBOSE=False,
                  restrict=True,
                  nsamples = False,
                  random=False):
        '''set the poles and samples such that the samples contain features in poles

        Args:
          steps (int): number of steps to qsample
          POLEFILE (str): file containing poles samples and features
          pole_1 (str): column name for first pole to use
          pole_2 (str): column name for second pole to use
          mutable (bool): Whether or not to set poles as the only mutable_vars
          VERBOSE (bool): boolean flag prints number of pole features not found in sample features if True
          restrict (bool): boolean flag restricts the sample features to polar features if True
          random (bool): boolean flag takes random sample of all_samples
        '''
        invalid_count = 0
        if all(x is not None for x in [self.samples, self.qnet]):
            poles = pd.read_csv(POLEFILE, index_col=0)
            self.poles=poles.transpose()
            self.polar_features = pd.concat([self.features, self.poles], axis=0).fillna('')
            poles_dict = {}
            for column in poles:
                p_ = self.polar_features.loc[column][self.cols].fillna('').values.astype(str)[:]
                poles_dict[column] = self.qsampling(p_,steps)
            self.poles_dict = poles_dict
            self.pL = self.poles_dict[pole_1]
            self.pR = self.poles_dict[pole_2]
            # self.pL = list(poles_dict.values())[0]
            # self.pR = list(poles_dict.values())[1]
            self.d0 = qdistance(self.pL, self.pR, self.qnet, self.qnet)
            
            if restrict:
                # restrict sample columns to polar columns
                cols = [x for x in self.poles.columns if x in self.samples.columns]
                self.samples=self.samples[cols]
                self.restricted = True
                
            elif self.restricted:
                # if poles had been restricted before, unrestrict it
                self.restricted = False
                self.samples = self.all_samples
                if self.nsamples is not None:
                    self.set_nsamples(nsamples, random)

            # for x in self.poles.columns:
            #     if x not in self.samples.columns:
            #         invalid_count += 1
            #         self.samples[x]=''

            self.samples = pd.concat([self.features,self.samples], axis=0).fillna('')
            #self.all_samples = self.samples
            self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
            
            if mutable:
                self.mutable_vars=[x for x in self.cols if x in self.poles.columns]
        elif self.samples is None:
            raise ValueError("load_data first!")

        if VERBOSE:
            print("{} pole features not found in sample features".format(invalid_count))

    def distance(self,
                sample1,
                sample2,
                nsteps1=0,
                nsteps2=0):
        """qsamples each sample set num of steps, then takes qdistance

        Args:
          sample1 (list[str]): sample vector 1, must have the same num of features as the qnet
          sample2 (list[str]): sample vector 2, must have the same num of features as the qnet
          nsteps1 (int, optional): number of steps to qsample for sample1
          nsteps2 (int, optional): number of steps to qsample for sample2

        Returns:
          qdistance: float, distance between two samples
        """
        if self.qnet is None:
            raise ValueError("load qnet first!")
        bp1 = self.getBaseFrequency(sample1)
        bp2 = self.getBaseFrequency(sample2)
        sample1 = qsample(sample1, self.qnet, nsteps1)#, baseline_prob=bp1)
        sample2 = qsample(sample2, self.qnet, nsteps2)#, baseline_prob=bp2)
        return qdistance(sample1, sample2, self.qnet, self.qnet)
    
    def __distfunc(self, 
                x, 
                y):
        '''Compute distance between two samples

        Args:
          x (list[str]): first sample
          y (list[str]): second sample
          
        Returns:
         d: qdistance
        '''
        d=qdistance(x,y,self.qnet,self.qnet)
        return d
    
    def polarDistance(self,
                    i,
                    return_dict=None):
        """return the distance from a sample to the poles

        Args:
          i (int): index of sample to take
          return_dict (dict): dict used for multiple sample function

        Returns:
          distances: float, distance from sample to each pole
        """
        samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
        p = samples_as_strings[i]
        distances = []
        for index, row in self.polar_features[self.cols].iterrows():
            row = row.fillna('').values.astype(str)[:]
            distances.append(self.distance(p, np.array(row)))
        if return_dict is not None:
            return_dict[i] = distances
        return distances
            
    def polarDistance_multiple(self,
                            outfile):
        """return the distance from all samples to the poles

        Args:
          outfile (str): desired output filename and path
          
        Returns:
          return_dict: dictionary containing multiprocessing results
        """
        if all(x is not None for x in [self.samples, self.cols,
                                    self.polar_features]):
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []
            
            for i in range(len(self.samples)):
                p = mp.Process(target=self.polarDistance, args=(i, return_dict))
                processes.append(p)

            [x.start() for x in processes]
            [x.join() for x in processes]

            pole_names = []
            for index, row in self.polar_features[self.cols].iterrows():
                pole_names.append(index)
            result=[x for x in return_dict.values()]
            result=pd.DataFrame(result,columns=pole_names).to_csv(outfile)
            
        else:
            raise ValueError("load data first!")
        return return_dict
        
    def distfunc_line(self,
                    i,
                    return_dict=None):
        '''compute the dist for a row, or vector of samples

        Args:
          i (int): row
        
        Return:
          line: float, numpy.ndarray
        '''
        if all(x is not None for x in [self.samples, self.features]):
            w = self.samples.index.size
            line = np.zeros(w)
            y = self.samples_as_strings[i]
            for j in range(w):
                # only compute half of the distance matrix
                if j > i:
                    x = self.samples_as_strings[j]
                    line[j] = self.__distfunc(x, y)
        else:
            raise ValueError("load_data first!")
        if return_dict is not None:
            return_dict[i] = line
        return line
    
    def distfunc_multiples(self,
                        outfile):
        """compute distance matrix for all samples in the dataset

        Args:
          outfile (str): desired output filename and path
          
        Returns:
          return_dict: dictionary containing multiprocessing results
        """
        if all(x is not None for x in [self.samples, self.features]):
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []

            for i in range(len(self.samples)):
                p = mp.Process(target=self.distfunc_line, args=(i, return_dict))
                processes.append(p)
            
            [x.start() for x in processes]
            [x.join() for x in processes]
            result=[x for x in return_dict.values()]
            columns = [i for i in range(len(self.samples))]
            result=pd.DataFrame(result,columns=columns, index=columns).sort_index(ascending=False)
            result = result.to_numpy()
            result = pd.DataFrame(np.maximum(result, result.transpose()))
            result.to_csv(outfile)
        else:
            raise ValueError("load data first!")
        
        return result
    
    def polar_separation(self,
                        nsteps=0):
        """calculates the distance between poles as a qdistance matrix

        Args:
          nsteps (int, optional): [description]. Defaults to 0.
          
        Returns:
          self.polar_matrix: dictionary containing multiprocessing results
        """
        polar_arraydata = self.polar_features[self.cols].values.astype(str)[:]
        samples_ = []
        for vector in polar_arraydata:
            bp = self.getBaseFrequency(vector)
            sample = qsample(vector, self.qnet, nsteps, baseline_prob=bp)
            samples_.append(sample)
        samples_ = np.array(samples_)
        self.polar_matrix = qdistance_matrix(samples_, samples_, self.qnet, self.qnet)
        return self.polar_matrix
        
    def embed(self,
            infile,
            name_pref,
            out_dir,
            pca_model=False,
            EMBED_BINARY=None):
        '''
        embed data

        Args:
          infile (str): input file to be embedded
          name_pref (str): preferred name for output file
          out_dir (str): output dir for results
          pca_model (bool): whether or not to generate PCA model
          EMBED_BINARY (os.path.abspath): path to embed binary
        '''
        if all(x is not None for x in [self.year]):
            yr = self.year
            PREF = name_pref
            FILE = infile

            if EMBED_BINARY is None:
                EMBED = pkgutil.get_data("cognet.bin", "__embed__.so") 
            else:
                EMBED = EMBED_BINARY
            DATAFILE = out_dir + 'data_' +yr
            EFILE = out_dir + PREF + '_E_' +yr
            DFILE = out_dir + PREF + '_D_' +yr

            pd.read_csv(FILE, header=None).to_csv(DATAFILE,sep=' ',header=None,index=None)
            STR=EMBED+' -f '+DATAFILE+' -E '+EFILE+' -D '+DFILE
            subprocess.call(STR,shell=True)
            if pca_model:
                embed_to_pca(EFILE, EFILE+'_PCA')
        elif self.year is None:
            raise ValueError("load_data first!")
    
    def __calc_d0(self,
                pole_1,
                pole_2):
        """calculate distance between two poles

        Args:
          pole_1 (list[str]): a polar vector, must have same number of features as qnet
          pole_2 (list[str]): a polar vector, must have same number of features as qnet
        """
        self.pL = self.poles_dict[pole_1]
        self.pR = self.poles_dict[pole_2]
        self.d0 = qdistance(self.pL, self.pR, self.qnet, self.qnet)
        
    def ideology(self,
                i,
                return_dict=None,
                pole_1=None,
                pole_2=None):
        """return ideology index (left-leaning or right-leaning) for a singular sample

        Args:
          i (int): index of sample
          pole_1 (int): index of Pole One to calc as base distance. Defaults to 0.
          pole_2 (int): index of Pole Two to calc as base distance. Defaults to 1.
          return_dict (dict, optional): dict containing results
          
        Returns:
          [ideology_index, dR, dL, self.d0]: which way the sample leans,
                                             distance from the right pole,
                                             distance from the left pole,
                                             and distance between poles, respectively
        """
        if pole_1 is not None or pole_2 is not None:
            self.__calc_d0(pole_1, pole_2)
            
        p = self.samples_as_strings[i]
        dR = qdistance(self.pR, p, self.qnet, self.qnet)
        dL = qdistance(self.pL, p, self.qnet, self.qnet)
        ideology_index = (dR-dL)/self.d0
        if return_dict is not None:
            return_dict[i] = [ideology_index, dR, dL, self.d0]
        return [ideology_index, dR, dL, self.d0]

    def dispersion(self,
                i,
                return_dict=None):
        """qsamples a sample n times and takes distance matrix 
        to determine max and std of distances between qsamples

        Args:
          i (int): index of sample
          return_dict (dict): dict containing results

        Returns:
          list[float]: std and max of the distances btwn qsamples
        """
        p = self.samples_as_strings[i]
        Qset = [qsample(p, self.qnet, self.steps) for j in np.arange(self.num_qsamples)]
        Qset = np.array(Qset)

        matrix = (qdistance_matrix(Qset, Qset, self.qnet, self.qnet))
        Q = matrix.max()
        Qsd = matrix.std()
        if return_dict is not None:
            return_dict[i] = [Qsd, Q]
        return [Qsd, Q]
    
    def compute_DLI_samples(self,
                        type,
                        outfile,
                        num_qsamples=40,
                        steps=120,
                        n_jobs=28,
                        pole_1=0,
                        pole_2=1):
        """compute and save ideology index or dispersion for all samples

        Args:
          num_qsamples (int): number of qsamples to compute
          outfile (str): output file for results
          type (str): whether to calc dispersion or ideology
          steps (int): number of steps to qsample
          n_jobs (int, optional): sets the number of jobs for parallelization. Defaults to 28.
          pole_1 (int, optional): index of Pole One to calc as base distance. Defaults to 0.
          pole_2 (int, optional): index of Pole Two to calc as base distance. Defaults to 1.

        Raises:
          ValueError: set poles if poles are not set
          ValueError: load data if samples or features are not present
            
        Returns:
          Result: dictionary containing multiprocessing results
        """
        if all(x is not None for x in [self.samples, self.features,
                                    self.pL, self.pR]):
            self.num_qsamples = num_qsamples
            self.steps = steps
            if pole_1 != 0 or pole_2 != 1:
                self.__calc_d0(pole_1, pole_2)
            
            # testing
            # pd.DataFrame(self.samples_as_strings).to_csv('examples_results/class_allsamples_2018.csv')
            
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []

            if type == 'ideology':
                for i in range(len(self.samples)):
                    p = mp.Process(target=self.ideology, args=(i, return_dict))
                    processes.append(p)
                columns=['ideology', 'dR', 'dL', 'd0']
            elif type == 'dispersion':
                for i in range(len(self.samples)):
                    p = mp.Process(target=self.dispersion, args=(i, return_dict))
                    processes.append(p)
                columns=['Qsd', 'Qmax']
            else:
                raise ValueError("Type must be either dispersion or ideology!")
            
            [x.start() for x in processes]
            [x.join() for x in processes]
            result=[x for x in return_dict.values()]
            result=pd.DataFrame(result,columns=columns).to_csv(outfile)

        elif self.pL is None or self.pR is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")
        return pd.DataFrame(return_dict.copy())

    def compute_polar_indices(self,
                            num_samples = None,
                            polar_comp = False,
                            POLEFILE = None,
                            steps = 5):
        '''set up polar indices for dissonance func

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
            
            polar_features = pd.concat([self.features, self.poles], axis=0)
            self.polar_indices=np.where(polar_features[self.cols].fillna('XXXX').values[0]!='XXXX')[0]
        
        elif self.poles is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")

    def dissonance(self,
                sample_index,
                return_dict=None,
                MISSING_VAL=0.0):
        '''compute dissonance for each sample_index, helper function for all_dissonance
        
        Args:
          sample_index (int): index of the sample to compute dissonance
          return_dict (dict): dict containing results
          MISSING_VAL (float): default dissonance value
          
        Returns: 
          diss: ndarray containing dissonance for sample
        '''
        if all(x is not None for x in [self.samples, self.features]):
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
            if return_dict is not None:
                return_dict[sample_index] = diss[self.polar_indices]
            return diss[self.polar_indices]
        else:
            raise ValueError("load_data first!")
    
    def dissonance_matrix(self,
                        output_file='/example_results/DISSONANCE_matrix.csv',
                        n_jobs=28):
        '''get the dissonance for all samples

        Args:
          output_file (str): directory and/or file for output
          n_jobs (int): number of jobs for pdqm

        Returns:
          pandas.DataFrame
        '''
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for i in range(len(self.samples)):
            p = mp.Process(target=self.dissonance, args=(i, return_dict))
            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]

        result=[x for x in return_dict.values()]
        if self.polar_indices is not None:
            polar_features = pd.concat([self.features, self.poles], axis=0)
            cols = polar_features[self.cols].dropna(axis=1).columns
        else:
            cols = self.cols
        result=pd.DataFrame(result,columns=cols).to_csv(output_file)
        
        self.dissonance_file = output_file
        return pd.DataFrame(return_dict.copy())
    
    def __choose_one(self,
                X):
        '''returns a random element of X

        Args:
          X (1D array-like): vector from which random element is to be chosen
        
        Returns:
          X: random element of sample
          None: if X has len 0
        '''
        X=list(X)
        if len(X)>0:
            return X[np.random.randint(len(X))]
        return None

    def getMaskedSample(self,
                        s,
                        mask_prob=0.5,
                        allow_all_mutable=False):
        '''inputs a sample and randomly mask elements of the sample

        Args:
          s (list[str]): vector of sample, must have the same num of features as the qnet
          mask_prob (float): float btwn 0 and 1, prob to mask element of sample
          allow_all_mutable (bool): whether or not all variables are mutable
          
        Returns:
          s1,base_frequency,MASKrand,
          np.where(base_frequency)[0],
          np.mean(rnd_match_prob),
          np.mean(max_match_prob),
          s_rand
        '''
        if self.samples is not None:
            s0=s.copy()
            s0=np.array(s0)   
            MUTABLE=pd.DataFrame(np.zeros(len(self.cols)),index=self.cols).transpose()
            WITHVAL=[x for x in self.cols[np.where(s0)[0]] if x in self.mutable_vars ]
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

    def randomMaskReconstruction(self,
                                index=None,
                                return_dict=None,
                                sample=None):
        """reconstruct the masked sample by qsampling and comparing to original
        set self.mask_prob and self.steps if needed

        Args:
          return_dict (dict): dict containing results. Defaults to None.
          sample (list[str], optional): sample vector, must have the same num of features as the qnet. Defaults to None.
          index (int): index of sample to take. Defaults to None.

        Raises:
          ValueError: Neither sample or index were given
          ValueError: Both sample and index were given
        """
        if all(x is None for x in [sample, index]):
            raise ValueError("Must input either sample or index!")
        elif all(x is not None for x in [sample, index]):
            raise ValueError("Must input either sample or index not both!")
        elif sample is not None:
            s=np.array(pd.DataFrame(sample).fillna('').values.astype(str)[:])
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

        cmpf=pd.DataFrame([s,qs,s_rand],columns=self.cols,index=['s','q','r'])[mask_].transpose()
        cmpf.index.name='gssvar'
        cmpf.to_csv('examples_results/CMPF_2018/CMPF-'+str(index)+'.csv')
        return_dict[index] = (1 - (dqestim/dactual))*100,rmatch_u,rmatch
        return (1 - (dqestim/dactual))*100,rmatch_u,rmatch,s,qs,s_rand,mask_

    def randomMaskReconstruction_multiple(self,
                                          out_file):
        '''runs and saves the results of the predicted masked sample

        Args:
          output_file (str): directory and/or file for output
          
        Returmns:
          result.rederr.mean(), result.rand_err.mean(): mean of reconstruction error and random error
        '''
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for i in range(len(self.samples)):
            p = mp.Process(target=self.randomMaskReconstruction, args=(i, return_dict))
            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]
        
        #result=pd.DataFrame(return_dict.items())[1]#, columns=['sample','rederr','r_prob','rand_err','s','q','r'])
        result=[x for x in return_dict.values() if isinstance(x, tuple)]
        # #result=pd.DataFrame(result.tolist())
        # print(result)
        # cmprdf=result[[3,4,5]]
        # mask_=result[[6]]
        # cmprdf.columns=['s','q','r']#[mask_].transpose()
        # cmprdf.to_csv("examples_results/CMPF_"+"tmp"+".csv")
        # print(cmprdf)
        # result=result[[0,1,2]]
        result=pd.DataFrame(result,columns=['rederr','r_prob','rand_err'])
        result.rederr=result.rederr.astype(float)

        if self.poles is not None:
            result.to_csv(out_file)
        else:
            result.to_csv(out_file)
        
        return pd.DataFrame(return_dict.copy())
    
    def dmat_filewriter(self,
                        pyfile,
                        QNETPATH,
                        MPI_SETUP_FILE="mpi_setup.sh",
                        MPI_RUN_FILE="mpi_run.sh",
                        MPI_LAUNCHER_FILE="mpi_launcher.sh",
                        YEARS='2016',
                        NODES=4,
                        T=12,
                        num_samples=None,
                        OUTFILE='tmp_distmatrix.csv',
                        tmp_samplesfile="tmp_samples_as_strings.csv"):
        if all(x is not None for x in [self.poles_dict,self.features,
                                       self.qnet, self.cols]):
            if num_samples is not None:
                self.set_nsamples(num_samples)
            
            tmp_path = "mpi_tmp/"
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            
            pd.DataFrame(self.samples_as_strings).to_csv(tmp_path+tmp_samplesfile, header=None, index=None)
            
            w = self.samples.index.size
            with open(tmp_path+pyfile, 'w+') as f:
                f.writelines(["from mpi4py.futures import MPIPoolExecutor\n",
                              "import numpy as np\n",
                              "import pandas as pd\n",
                              "from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix\n",
                              "from quasinet.qsampling import qsample, targeted_qsample\n\n",
                              "qnet=load_qnet(\'{}\')\n".format(QNETPATH)])

                f.writelines(["w = {}\n".format(w),
                              "h = w\n",
                              "p_all = pd.read_csv(\"{}\", header=None)\n\n".format(tmp_samplesfile)])

                f.writelines(["def distfunc(x,y):\n",
                              "\td=qdistance(x,y,qnet,qnet)\n",
                              "\treturn d\n\n"])

                f.writelines(["def dfunc_line(k):\n",
                              "\tline = np.zeros(w)\n",
                              "\ty = np.array(p_all.iloc[k])\n",
                              "\tfor j in range(w):\n",
                              "\t\tif j > k:\n",
                              "\t\t\tx = np.array(p_all.iloc[j])\n",
                              "\t\t\tline[j] = distfunc(x, y)\n",
                              "\treturn line\n\n"])

                f.writelines(["if __name__ == '__main__':\n",
                              "\twith MPIPoolExecutor() as executor:\n",
                              "\t\tresult = executor.map(dfunc_line, range(h))\n",
                              "\tresult = pd.DataFrame(result)\n",
	                          "\tresult = result.to_numpy()\n",
                              "\tresult = pd.DataFrame(np.maximum(result, result.transpose()))\n"
                              "\tresult.to_csv(\'{}\',index=None,header=None)".format(OUTFILE)])
                
            with open(tmp_path+MPI_SETUP_FILE, 'w+') as ms:
                ms.writelines(["#!/bin/bash\n",
                               "YEAR=$1\n\n",
                               "if [ $# -gt 1 ] ; then\n",
                               "\tNODES=$2\n",
                               "else\n",
                               "\tNODES=3\n",
                               "fi\n",
                               "if [ $# -gt 2 ] ; then\n",
                               "\tNUM=$3\n",
                               "else\n",
                               "\tNUM='all'\n",
                               "fi\n",
                               "if [ $# -gt 3 ] ; then\n",
                               "\tPROG=$4\n",
                               "else\n",
                               "\tPROG=$(tty)\n",
                               "fi\n\n",
                               "NUMPROC=`expr 28 \* $NODES`\n",
                               "echo \"module load midway2\" >> $PROG\n",
                               "echo \"module unload python\" >> $PROG\n",
                               "echo \"module unload openmpi\" >> $PROG\n",
                               "echo \"module load python/anaconda-2020.02\" >> $PROG\n",
                               "echo \"module load mpi4py\" >> $PROG\n",
                               "echo \"date; mpiexec -n \"$NUMPROC\" python3 -m mpi4py.futures {}; date\"  >> $PROG\n".format(pyfile),
                                ])

            with open(tmp_path+MPI_RUN_FILE, 'w+') as mr:
                mr.writelines(["#!/bin/bash\n",
                               "YEARS=\'{}\'\n".format(YEARS),
                               "# nodes requested\n",
                               "NODES={}\n".format(NODES),
                               "# time requested\n",
                               "T={}\n".format(T),
                               "NUM=\'all\'\n",
                               "LAUNCH=\'../mpi_launcher.sh\'\n\n",
                               "for yr in `echo $YEARS`\n",
                               "do\n",
                               "\techo $yr\n",
                               "\t./{} $yr $NODES $NUM tmp_\"$yr\"\n".format(MPI_SETUP_FILE),
                               "\t$LAUNCH -P tmp_\"$yr\" -F -T $T -N \"$NODES\" -C 28 -p broadwl -J ACRDALL_\"$yr\" -M 56\n",
                               "done\n",
                               "rm tmp_\"$yr\"*\n"])
        else:
            raise ValueError("load data first!")
