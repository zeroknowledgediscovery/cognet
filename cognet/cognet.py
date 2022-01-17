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

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pqdm.threads import pqdm  

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
        self.s_null = None
        self.D_null = None
        self.mask_prob = 0.5
        self.variation_weight = None
        self.polar_matrix = None
        self.nsamples = None
        self.restricted = False
        self.MAX_PROCESSES = 0
    
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
            # inherit atrributes from model object
            self.qnet = model.myQnet
            featurenames, samples = data_obj.format_samples(key)
            samples = pd.DataFrame(samples)
            self.cols = np.array(featurenames)
            self.features = pd.DataFrame(columns=np.array(featurenames))
            
            # inherit mutable and immutable variables from model obj
            if any(x is not None for x in [model.immutable_vars, model.mutable_vars]):
                if model.immutable_vars is not None:
                    self.immutable_vars = model.immutable_vars
                    self.mutable_vars = [x for x in self.features if x not in self.immutable_vars]
                elif model.mutable_vars is not None:
                    self.mutable_vars = model.mutable_vars
                    self.immutable_vars = [x for x in self.features if x not in self.mutable_vars]
            else:
                self.mutable_vars = self.features
            
            # inherit and set class attributes.
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
        """read in either train or test data, specified by key, from data obj,
        and inherit other attributes.

        Args:
          data_obj (class): instance of dataformatter class
          key (str): 'all', 'train', or 'test', corresponding to sample type
          
        Returns:
          featurenames, samples: formatted arrays
        """
        # inherit attributes from dataformatter object
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
                  Qnet):
        '''load cols, features, samples, and qnet.

        Args:
          year (str): to identify cols/features.
          features_by_year (str): file containing all features by year of the dataset.
          samples (str): file of samples for that year.
          Qnet (str): Qnet file location.
        '''
        # set attributes from given files and data
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

        # read in samples and initialize related attributes
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
        # set mutable and immutable variable attributes 
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
        # each time function is called, reset samples to use_all_samples
        # this allows us to call nsamples numerous times 
        self.samples = self.all_samples
        if self.samples is not None:
            # if a greater number of sample is selected than available, raise error
            if all(x is not None for x in [num_samples, self.samples]):
                if num_samples > len(self.samples.index):
                    string = 'The number of selected samples ({}) ' + \
                        'is greater than the number of samples ({})!'
                    string = string.format(num_samples, len(self.samples.index))
                    raise ValueError(string)

                # if the same number of samples is selected as available, print warning
                if num_samples == len(self.samples.index):
                    string = 'The number of selected samples ({}) ' + \
                        'is equal to the number of samples ({})!'
                    string = string.format(num_samples, len(self.samples.index))
                    print(string)
                    
                # if random is true, return random sample, otherwise return an ordered slice
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
        # if variable is not mutable, set its base frequency to zero 
        MUTABLE=pd.DataFrame(np.zeros(len(self.cols)),index=self.cols).transpose()
             
        for m in self.mutable_vars:
            MUTABLE[m]=1.0
        mutable_x=MUTABLE.values[0]
        base_frequency=mutable_x/mutable_x.sum()
        
        # otherwise, set base frequency weighted by variation weight
        for i in range(len(base_frequency)):
            if base_frequency[i]>0.0:
                base_frequency[i]= self.variation_weight[i]*base_frequency[i]

        return base_frequency/base_frequency.sum()
    
    def qsampling(self,
                sample,
                steps,
                immutable=False):
        '''perturb the sample based on the qnet distributions and number of steps

        Args:
          sample (1d array-like): sample vector, must have the same num of features as the qnet
          steps (int): number of steps to qsample
          immutable (bool): are there variables that are immutable?
        '''
        # immutable, check that mutable variables have been initialized
        if immutable == True:
            if all(x is not None for x in [self.mutable_vars, sample]):
                return qsample(sample,self.qnet,steps,self.getBaseFrequency(self.samples))
            elif self.mutable_vars is None:
                raise ValueError("set mutable and immutable variables first!")
        else:
            return qsample(sample,self.qnet,steps)

    def random_sample(self,
                      type="prob",
                      df=None,
                      n=1,
                      steps=200,
                      n_jobs=3):
        '''compute a random sample from the underlying distributions of the dataset, by column.
        
        
        Args:
          type (str): How to randomly draw samples. Can take on "null", "uniform", or "prob". Deafults to "prob".
          df (pandas.DataFrame): Desired data to take random sample of. Defaults to None, in which case qnet samples are used.
          n (int): number of random samples to take. Defaults to 1.
          steps (int): number of steps to qsample. Defaults to 1000
          
        Returns:
          return_df (pd.DataFrame): Drawn random sample.
        '''
        # check if a new dataset was given
        if df is None:
            samples_ = self.samples
        else:
            samples_ = df

        return_df = pd.DataFrame()
        # take random sample from each of the columns based on their probability distribution
        if type == "prob":
            for col in samples_.columns:
                return_df[col] = samples_[col].sample(n=n, replace=True).values
                
        # random sampling using Qnet qsampling
        elif type == "null":
            null_array = np.zeros((len(samples_.columns),), dtype=str)
            args = [[null_array, steps] for i in range(n)]
            qsamples = pqdm(args, self.qsampling, n_jobs=n_jobs, argument_type='args') 
            
            # for i in range(n):
            #     qsamples.append(self.qsampling(null_array, steps))
            return_df = pd.DataFrame(qsamples, columns=samples_.columns)
            
        # random sampling using uniform distribution of values by Columns
        elif type == "uniform":
            for col in samples_.columns:
                # get unqiue values for each column and draw n values randomly
                values = samples_[col].unique().astype(str)
                return_df[col]=np.random.choice(values, size=n, replace=True)
        else:
            raise ValueError("Type is not supported!")
        return return_df
    
    def set_poles(self,
                  POLEFILE,
                  pole_1,
                  pole_2,
                  steps=0,
                  mutable=False,
                  VERBOSE=False,
                  restrict=False,
                  nsamples = None,
                  random=False):
        '''set the poles and samples such that the samples contain features in poles

        Args:
          steps (int): number of steps to qsample
          POLEFILE (str): file containing poles samples and features
          pole_1 (str): column name for first pole
          pole_2 (str): column name for second pole
          mutable (bool): Whether or not to set poles as the only mutable_vars
          VERBOSE (bool): boolean flag prints number of pole features not found in sample features if True
          restrict (bool): boolean flag restricts the sample features to polar features if True. Defaults to False.
          random (bool): boolean flag takes random sample of all_samples
        '''
        invalid_count = 0
        if all(x is not None for x in [self.samples, self.qnet]):
            # read and set poles
            poles = pd.read_csv(POLEFILE, index_col=0)
            self.poles=poles.transpose()
            self.polar_features = pd.concat([self.features, self.poles], axis=0).fillna('')
            poles_dict = {}
            for column in poles:
                p_ = self.polar_features.loc[column][self.cols].fillna('').values.astype(str)[:]
                # qsample poles to qnet
                poles_dict[column] = self.qsampling(p_,steps)
            self.poles_dict = poles_dict
            self.pL = self.poles_dict[pole_1]
            self.pR = self.poles_dict[pole_2]
            self.d0 = qdistance(self.pL, self.pR, self.qnet, self.qnet)
            
            # restrict sample columns to polar columns
            if restrict:
                cols = [x for x in self.poles.columns if x in self.samples.columns]
                self.samples=self.samples[cols]
                self.restricted = True
                self.samples = pd.concat([self.features,self.samples], axis=0).replace("nan","").fillna('')
                self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
                
            # if restrict==False, unrestrict it and set original
            else:
                self.restricted = False
                self.samples = self.all_samples
                if self.nsamples is not None:
                    self.set_nsamples(nsamples, random)
            
            # identify pole features that were excluded due to sample features restriction
            if VERBOSE:
                for x in self.poles.columns:
                    if x not in self.samples.columns:
                        invalid_count += 1
                        #self.samples[x]=''
            
            if mutable:
                self.mutable_vars=[x for x in self.cols if x in self.poles.columns]
        elif self.samples is None:
            raise ValueError("load_data first!")

        if VERBOSE:
            print("{} pole features not found in sample features".format(invalid_count))

    def mp_compute(self, 
                   processes,
                   func, 
                   cols,
                   outfile, 
                   args=[]):
        """
        Compute desired function through multiprocessing and save result to csv.

        Args:
          processes (int): number of processes to use.
          func (func): function to compute using multiprocessing
          cols (list): column names of resulting csv
          outfile (str)): filepath + filename for resulting csv
          args (list): list containing arguments for desired function. Defaults to empty list.
        """

        # init mp.Manager and result dict
        manager = mp.Manager()
        return_dict = manager.dict()

        # set processes as given, unless class parameter is set
        max_processes = processes
        if self.MAX_PROCESSES != 0:
            max_processes = self.MAX_PROCESSES
            print("Number of Processes {} has been set using class parameter".format(self.MAX_PROCESSES))
        num_processes = 0
        process_list = []
        
        # init mp.Processes for each individual sample
        # run once collected processes hit max
        for i in range(len(self.samples)):
            params = tuple([i, return_dict] + args)
            num_processes += 1
            p = mp.Process(target=func,
                        args=params)
            process_list.append(p)
            if num_processes == max_processes:
                [x.start() for x in process_list]
                [x.join() for x in process_list]
                process_list = []
                num_processes = 0
                
        # compute remaining processes
        if num_processes != 0:
            [x.start() for x in process_list]
            [x.join() for x in process_list]
            process_list = []
            num_processes = 0
        
        # format and save resulting dict
        result = pd.DataFrame(return_dict.values(), columns=cols, index=return_dict.keys()).sort_index()
        result.to_csv(outfile, index=None)
        return result
    
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
        #bp1 = self.getBaseFrequency(sample1)
        #bp2 = self.getBaseFrequency(sample2)
        # qsample samples
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
    
    def distfunc_line(self,
                    i,
                    return_dict=None):
        '''compute the distance for a single sample against all other samples

        Args:
          i (int): row
          return_dict (dict): dictionary containing multiprocessing results
        
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
                           outfile,
                           processes=6,
                           samples=None):
        """compute distance matrix for all samples in the dataset

        Args:
          outfile (str): desired output filename and path
          processes (int): Number of processes to run in parallel. Defaults to 6.
          samples (2D array): Dataset from which to calculate qdist matrix. Defaults to None.
          
        Returns:
          result: pandas.DataFrame containing distance matrix
        """
        if all(x is not None for x in [self.samples, self.features]):
            # if exterior dataset is given, temporarily replace class attributes
            if samples is not None:
                original_samples = self.samples
                original_samples_as_strings = self.samples_as_strings
                self.samples = samples
                samples = samples.fillna("").values.astype(str)
                self.samples_as_strings = samples
            cols = [i for i in range(len(self.samples))]
            result = self.mp_compute(processes,
                                        self.distfunc_line,
                                        cols,
                                        outfile)
            
            # format and save resulting dict, and tranpose symmetrical distance matrix
            result = result.to_numpy()
            result = pd.DataFrame(np.maximum(result, result.transpose()))
            result.to_csv(outfile, index=None, header=None)
            
            # replace class attributes with originals
            if samples is not None:
                self.samples = original_samples
                self.samples_as_strings = original_samples_as_strings
        else:
            raise ValueError("load data first!")
        
        return result
    
    def polarDistance(self,
                    i,
                    return_dict=None):
        """return the distances from a single sample to the poles

        Args:
          i (int): index of sample to take
          return_dict (dict): dictionary containing multiprocessing results

        Returns:
          distances: float, distance from sample to each pole
        """
        p = self.samples_as_strings[i]
        distances = []
        # calculate from each pole to the sample, and append to array
        for index, row in self.polar_features[self.cols].iterrows():
            row = row.fillna('').values.astype(str)[:]
            distances.append(self.distance(p, np.array(row)))
        if return_dict is not None:
            return_dict[i] = distances
        return distances
            
    def polarDistance_multiple(self,
                               outfile,
                               processes=6):
        """return the distance from all samples to the poles

        Args:
          outfile (str): desired output filename and path
          
        Returns:
          result: pandas.DataFrame containing polar distance results
        """
        if all(x is not None for x in [self.samples, self.cols,
                                    self.polar_features]):
            # get the column names
            pole_names = []
            for index, row in self.polar_features[self.cols].iterrows():
                pole_names.append(index)
            result = self.mp_compute(processes,
                                        self.polarDistance,
                                        pole_names,
                                        outfile)
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
        # vectorize and qsample poles
        polar_arraydata = self.polar_features[self.cols].values.astype(str)[:]
        samples_ = []
        for vector in polar_arraydata:
            bp = self.getBaseFrequency(vector)
            sample = qsample(vector, self.qnet, nsteps, baseline_prob=bp)
            samples_.append(sample)
        samples_ = np.array(samples_)
        # calculate distance matrix for poles
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
        # if all(x is not None for x in [self.year]):
            # init file names 
        yr = ''
        if self.year is not None:
            yr = self.year
        PREF = name_pref
        FILE = infile
        DATAFILE = out_dir + 'data_' +yr
        EFILE = out_dir + PREF + '_E_' +yr
        DFILE = out_dir + PREF + '_D_' +yr
        
        # set embed binary directory
        if EMBED_BINARY is None:
            EMBED = pkgutil.get_data("cognet.bin", "__embed__.so") 
        else:
            EMBED = EMBED_BINARY
        
        # embed data files
        pd.read_csv(FILE, header=None).to_csv(DATAFILE,sep=' ',header=None,index=None)
        STR=EMBED+' -f '+DATAFILE+' -E '+EFILE+' -D '+DFILE
        subprocess.call(STR,shell=True)
        if pca_model:
            embed_to_pca(EFILE, EFILE+'_PCA')
        # elif self.year is None:
        #    raise ValueError("load_data first!")
    
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
        # calculate base distance between two poles
        if pole_1 is not None or pole_2 is not None:
            self.__calc_d0(pole_1, pole_2)
        
        # calculate distances between sample and the two poles
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
          return_dict (dict): dictionary containing multiprocessing results

        Returns:
          list[float]: std and max of the distances btwn qsamples
        """
        # qsample sample num_qsample times
        p = self.samples_as_strings[i]
        Qset = [qsample(p, self.qnet, self.steps) for j in np.arange(self.num_qsamples)]
        Qset = np.array(Qset)

        # calculate qdistance matrix for qsampled samples
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
                        pole_2=1,
                        processes=6):
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
          result: pandas.DataFrame containing multiprocessing results
        """
        if all(x is not None for x in [self.samples, self.features,
                                    self.pL, self.pR]):
            # init vars
            self.num_qsamples = num_qsamples
            self.steps = steps
            if pole_1 != 0 or pole_2 != 1:
                self.__calc_d0(pole_1, pole_2)
            
            if type == 'ideology':
                func_ = self.ideology
                cols=['ideology', 'dR', 'dL', 'd0']
            elif type == 'dispersion':
                func_ = self.dispersion
                cols=['Qsd', 'Qmax']
            else:
                raise ValueError("Type must be either dispersion or ideology!")
            
            result = self.mp_compute(processes,
                                     func_,
                                     cols,
                                     outfile)
        elif self.pL is None or self.pR is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")
        return result

    def compute_polar_indices(self,
                              num_samples=None,
                              polar_comp=False,
                              POLEFILE=None,
                              steps=5):
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

            if polar_comp:
                self.set_poles(self.qnet, steps, POLEFILE)
            
            # calculate polar indices
            polar_features = pd.concat([self.features, self.poles], axis=0)
            self.polar_indices=np.where(polar_features[self.cols].fillna('XXXX').values[0]!='XXXX')[0]
        
        elif self.poles is None:
            raise ValueError("set_poles first!")
        else:
            raise ValueError("load_data first!")

    def dissonance(self,
                    sample_index=0,
                    return_dict=None,
                    MISSING_VAL=0.0,
                    sample=None):
        '''compute dissonance for a single sample, helper function for all_dissonance
        
        Args:
          sample_index (int): index of the sample to compute dissonance. Defaults to 0.
          return_dict (dict): dictionary containing multiprocessing results
          MISSING_VAL (float): default dissonance value
          sample (1D array): sample to compute dissonance of, instead of using sample index. Defaults to None.
          
        Returns: 
          diss[self.polar_indices]: ndarray containing dissonance for sample
        '''
        if all(x is not None for x in [self.samples, self.features]):
            if sample is None:
                s = self.samples_as_strings[sample_index]
            else:
                s = sample
            if self.polar_indices is None:
                self.polar_indices = range(len(s))

            # init vars and calculate dissonance for sample
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
                        outfile='/example_results/DISSONANCE_matrix.csv',
                        processes=6):
        '''get the dissonance for all samples

        Args:
          output_file (str): directory and/or file for output
          processes (int): max number of processes. Defaults to 6.

        Returns:
          result: pandas.DataFrame containing dissonances for each sample
        '''
        # set columns
        if self.polar_indices is not None:
            polar_features = pd.concat([self.features, self.poles], axis=0)
            cols = polar_features[self.cols].dropna(axis=1).columns
        else:
            cols = self.cols
        
        result = self.mp_compute(processes,
                                    self.dissonance,
                                    cols,
                                    outfile)
        return result
    
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
          s (list[str]): vector of sample, must have the same num of features as the qnet.
          mask_prob (float): float btwn 0 and 1, prob to mask element of sample. Defaults to 0.5
          allow_all_mutable (bool): whether or not all variables are mutable. Defaults to False.
          
        Returns:
          s1,
          base_frequency,
          MASKrand,
          np.where(base_frequency)[0],
          np.mean(rnd_match_prob),
          np.mean(max_match_prob),
          random_sample
        '''
        if self.samples is not None:
            # init random mutable variable masking
            s0=s.copy()
            s0=np.array(s0)   
            # double check, because code seems to imply that masking happens in order,
            # i.e. limited to the first 100 features, if there are only 100 mutable features
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

            # mask sample according to masking (base_frequency)
            s1=s.copy()
            for i in range(len(base_frequency)):
                if base_frequency[i]>0.0001:
                    s1[i]=''
                
            # create a random sample to test reconstruction effectiveness
            random_sample=np.copy(s)
            rnd_match_prob=[]        
            max_match_prob=[]        
            D=self.qnet.predict_distributions(s)
            for i in MASKrand:
                random_sample[np.where(
                    self.cols==i)[0][0]]=self.__choose_one(
                        self.D_null[np.where(self.cols==i)[0][0]].keys())
                    
                rnd_match_prob=np.append(rnd_match_prob,1/len(
                    self.D_null[np.where(self.cols==i)[0][0]].keys()))
                
                max_match_prob=np.append(
                    max_match_prob,np.max(
                        list(D[np.where(
                            self.cols==i)[0][0]].values())))
            
            # calculate base_frequency if all variables are mutable
            if allow_all_mutable:
                WITHVAL=[x for x in self.cols[np.where(s0)[0]]]
                MASKrand=[x for x in WITHVAL if random.random() < mask_prob ]
                for m in MASKrand:
                    MUTABLE[m]=1.0
                mutable_x=MUTABLE.values[0]
                base_frequency=mutable_x/mutable_x.sum()
                s1=s.copy()
                for i in range(len(base_frequency)):
                    if base_frequency[i]>0.0001:
                        s1[i]=''

            return s1,base_frequency,MASKrand,np.where(
                base_frequency)[0],np.mean(rnd_match_prob),np.mean(max_match_prob),random_sample
        else:
            raise ValueError("load_data first!")

    def randomMaskReconstruction(self,
                                index=None,
                                return_dict=None,
                                sample=None,
                                index_colname="feature_names",
                                output_dir="recon_results/",
                                file_name="recon_tmp.csv",
                                mask_prob=0.5,
                                allow_all_mutable=False,
                                save_samples=False,
                                save_output=True):
        """reconstruct the masked sample by qsampling and comparing to original
        set self.mask_prob and self.steps if needed

        Args:
          index (int): index of sample to take.
          return_dict (dict): dictionary containing multiprocessing results. Defaults to None.
          sample (list[str], optional): sample vector, must have the same num of features as the qnet. Defaults to None.
          index_colname (str): column name for index. Defaults to "feature_names"
          output_dir (str): directory name for output files. Defaults to "recon_results/".
          file_name (str): base file name for output files Defaults to "recon_tmp.csv".
          mask_prob (float): float btwn 0 and 1, prob to mask element of sample. Defaults to 0.5
          allow_all_mutable (bool): whether or not all variables are mutable. Defaults to False.
          save_samples (bool): whether to include sample vectors in the savefile. Defaults to False.
          save_output (bool): whether or not to save output df to file. Defaults to True.

        Raises:
          ValueError: Neither sample or index were given
          ValueError: Both sample and index were given
          
        Returns:
          return_values:(1 - (dqestim/dactual))*100,
                            rmatch_u,
                            rmatch,
                            s,
                            qs,
                            random_sample,
                            mask_
        """
        if all(x is None for x in [sample, index]):
            raise ValueError("Must input either sample or index!")
        elif all(x is not None for x in [sample, index]):
            raise ValueError("Must input either sample or index not both!")
        elif sample is not None:
            s=sample#np.array(pd.DataFrame(sample).fillna('').values.astype(str)[:])
        elif index is not None:
            s=self.samples_as_strings[index]
        
        # calculate masked sample and get variables
        s1,bp,mask_,maskindex,rmatch_u,rmatch,random_sample=self.getMaskedSample(s, 
                                                                        mask_prob=mask_prob,
                                                                        allow_all_mutable=allow_all_mutable)
        # if base_frequency is nan, set return_dict to nans
        if np.isnan(bp).any():
            return_dict[index] = np.nan,np.nan,np.nan
            return np.nan,np.nan,np.nan
        
        # make directories
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # qsample sample and calculate distances between original vs qsampled and masked
        qs=qsample(s1,self.qnet,self.steps,bp)
        dqestim=qdistance(s,qs,self.qnet,self.qnet)
        dmask=qdistance(s,s1,self.qnet,self.qnet)
        
        # format and save sample, qsample statistics and values
        cmpf=pd.DataFrame([s,qs,random_sample],
                          columns=self.cols,
                          index=['sample','qsampled','random_sample'])[mask_].transpose()
        cmpf.index.name= index_colname
        if save_output:
            file_name = file_name.replace("tmp", str(index))
            cmpf.to_csv(output_dir+file_name)
            
        if save_samples:
            return_values = (1 - (dqestim/dmask))*100,rmatch_u,rmatch,mask_,s,qs,random_sample
        else:
            return_values = (1 - (dqestim/dmask))*100,rmatch_u,rmatch,mask_
        
        if return_dict is not None:
            return_dict[index] = return_values
            return return_dict[index]
        return return_values

    def randomMaskReconstruction_multiple(self,
                                          outfile,
                                          steps=200,
                                          processes=6,
                                          index_colname="feature_names",
                                          output_dir="recon_results/",
                                          file_name="recon_tmp.csv",
                                          mask_prob=0.5,
                                          allow_all_mutable=False,
                                          save_samples=False,
                                          save_output=True):
        '''runs and saves the results of the predicted masked sample

        Args:
          output_file (str): directory and/or file for output.
          processes (int): max number of processes. Defaults to 6.
          index_colname="feature_names",
          output_dir="recon_results/",
          file_name="recon_tmp.csv",
          mask_prob (float): float btwn 0 and 1, prob to mask element of sample. Defaults to 0.5
          allow_all_mutable (bool): whether or not all variables are mutable. Defaults to False.
          save_samples (boolean): whether or not to save the generated qsamples, random samples, etc. Defaults to False.
          save_output (bool): whether or not to save output df to file. Defaults to True.
          
        Returns:
          result: pandas.DataFrame containing masking and reconstruction results.
        '''
        # set columns for mp_compute
        if save_samples:
            cols = ['rederr','r_prob','rand_err','mask_','sample','qsampled','random_sample']
        else:
            cols = ['rederr','r_prob','rand_err','mask_']
        
        # update class steps
        self.steps = steps
        
        # set args
        args=[None, index_colname, output_dir,
              file_name, mask_prob, allow_all_mutable, 
              save_samples, save_output]
        
        result = self.mp_compute(processes,
                                    self.randomMaskReconstruction,
                                    cols,
                                    outfile,
                                    args=args)
        return result
    
    def dmat_filewriter(self,
                        QNETPATH,
                        mpi_path="mpi_tmp/",
                        pyfile="cognet_qdistmatrix.py",
                        MPI_SETUP_FILE="mpi_setup.sh",
                        MPI_RUN_FILE="mpi_run.sh",
                        MPI_LAUNCHER_FILE="../launcher.sh",
                        YEARS='2016',
                        NODES=4,
                        T=12,
                        num_samples=None,
                        OUTFILE='tmp_distmatrix.csv',
                        tmp_samplesfile="tmp_samples_as_strings.csv"):
        """generate files to compute qdistance matrix using mpi parallelization

        Args:
          QNETPATH (str): Qnet filepath
          pyfile (str, optional): Name of generated python file. Defaults to "cognet_qdistmatrix.py".
          MPI_SETUP_FILE (str, optional): Name of mpi setup script. Defaults to "mpi_setup.sh".
          MPI_RUN_FILE (str, optional): Name of mpi run script. Defaults to "mpi_run.sh".
          MPI_LAUNCHER_FILE (str, optional): Launcher script filepath. Defaults to "launcher.sh".
          YEARS (str, optional): If looping by year, not currently implemented. Defaults to '2016'.
          NODES (int, optional): Number of nodes to use. Defaults to 4.
          T (int, optional): Number of hours to reserve nodes for. Defaults to 12.
          num_samples ([type], optional): How many samples to take. Defaults to None.
          OUTFILE (str, optional): CSV File to write computed qdist matrix. Defaults to 'tmp_distmatrix.csv'.
          tmp_samplesfile (str, optional): CSV File to write samples as strings. Defaults to "tmp_samples_as_strings.csv".

        Raises:
            ValueError: load data if qnet, features, or samples are not present]
        """
        if all(x is not None for x in [self.samples,self.features,
                                       self.qnet, self.cols]):
            if num_samples is not None:
                self.set_nsamples(num_samples)
            
            # init and make tmp dir 
            tmp_path = mpi_path
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            
            pd.DataFrame(self.samples_as_strings).to_csv(tmp_path+tmp_samplesfile, header=None, index=None)
            
            w = self.samples.index.size
            
            # writing python file
            with open(tmp_path+pyfile, 'w+') as f:
                f.writelines(["from mpi4py.futures import MPIPoolExecutor\n",
                              "import numpy as np\n",
                              "import pandas as pd\n",
                              "from quasinet.qnet import Qnet, qdistance, load_qnet, qdistance_matrix\n",
                              "from quasinet.qsampling import qsample, targeted_qsample\n\n",
                              "qnet=load_qnet(\'{}\')\n".format(QNETPATH)])

                f.writelines(["w = {}\n".format(w),
                              "h = w\n",
                              "p_all = pd.read_csv(\"{}\", header=None).values.astype(str)[:]\n\n".format(tmp_samplesfile)])

                f.writelines(["def distfunc(x,y):\n",
                              "\td=qdistance(x,y,qnet,qnet)\n",
                              "\treturn d\n\n"])

                f.writelines(["def dfunc_line(k):\n",
                              "\tline = np.zeros(w)\n",
                              "\ty = p_all[k]\n",
                              "\tfor j in range(w):\n",
                              "\t\tif j > k:\n",
                              "\t\t\tx = p_all[j]\n",
                              "\t\t\tline[j] = distfunc(x, y)\n",
                              "\treturn line\n\n"])

                f.writelines(["if __name__ == '__main__':\n",
                              "\twith MPIPoolExecutor() as executor:\n",
                              "\t\tresult = executor.map(dfunc_line, range(h))\n",
                              "\tresult = pd.DataFrame(result)\n",
	                          "\tresult = result.to_numpy()\n",
                              "\tresult = pd.DataFrame(np.maximum(result, result.transpose()))\n"
                              "\tresult.to_csv(\'{}\',index=None,header=None)".format(OUTFILE)])
            
            # writing MPI setup file
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

            # writing MPI run file
            with open(tmp_path+MPI_RUN_FILE, 'w+') as mr:
                mr.writelines(["#!/bin/bash\n",
                               "YEARS=\'{}\'\n".format(YEARS),
                               "# nodes requested\n",
                               "NODES={}\n".format(NODES),
                               "# time requested\n",
                               "T={}\n".format(T),
                               "NUM=\'all\'\n",
                               "LAUNCH=./\'{}\'\n\n".format(MPI_LAUNCHER_FILE),
                               "for yr in `echo $YEARS`\n",
                               "do\n",
                               "\techo $yr\n",
                               "\t./{} $yr $NODES $NUM tmp_\"$yr\"\n".format(MPI_SETUP_FILE),
                               "\t$LAUNCH -P tmp_\"$yr\" -F -T $T -N \"$NODES\" -C 28 -p broadwl -J MPI_TMP_\"$yr\" -M 56\n",
                               "done\n",
                               "rm tmp_\"$yr\"*\n"])
            os.system("cp {} {}".format(MPI_LAUNCHER_FILE,tmp_path+'mpi_launcher.sh'))
        
        else:
            raise ValueError("load data first!")