import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from cognet.util import assert_None, assert_array_dimension
class dataFormatter:
    """Aggregate related Qnet functions
    """

    def __init__(self,
                 samples):
        """init

        Args:
            samples ([str], optional): 2D array with rows as observations and columns as features.
        """
        self.samples = pd.read_csv(samples)
        self.features = {}
        self.nan_cols = []
        self.immutable_vars = None
        self.mutable_vars = None
        self.test_size = None
        self.random_state = None
        self.train_data = None
        self.test_data = None

    def __train_test_split(self,
                           test_size,
                           train_size=None,
                           random_state=None):
        """split the samples into training and testing samples

        Args:
          test_size (float): fraction of sample to take as test_size.
          train_size (float): fraction of sample to take as train_size. Defaults to None, and 1-test_size
          random_state (int, optional): random seed to split samples dataset . Defaults to None.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.train_data, self.test_data = train_test_split(self.samples,
                                                           test_size=test_size,
                                                           train_size=train_size,
                                                           random_state=random_state)
    
    def __Qnet_formatter(self,
                         samples,
                         key=None):
        """format data for Qnet input

        Args:
          samples ([str], optional): 2D array with rows as observations and columns as features.
          key (str): Either 'train' or 'test' key, to determine which set of features
        
        Returns:
            features and samples of either the train and test dataset
        """
        # if not isinstance(samples, np.ndarray):
        #     raise ValueError('Samples must be in numpy array form!')
        features = np.array(samples.columns.astype(str)[:])
        samples = samples.values.astype(str)[:]
        # remove columns that are all NaNs
        not_all_nan_cols = ~np.all(samples == '', axis=0)
        self.nan_cols = np.all(samples == '', axis=0)

        samples = samples[:, not_all_nan_cols]
        
        features = features[not_all_nan_cols]
        features = list(features)
        if key is not None:
            self.features[key] = features
        return features, samples

    def format_samples(self,
                       key):
        """formats samples and featurenames, either all, train, or test
        
        Args:
          key (str): 'all', 'train', or 'test', corresponding to sample type

        Returns:
            samples and featurenames: formatted
        """
        
        
        if all(x is not None for x in [self.train_data,
                                       self.test_data,
                                       self.samples]):
            raise ValueError("Split samples into test and train datasets or input samples first!")
        if key == 'train':
            samples = self.train_data
        elif key == 'test':
            samples = self.test_data
        elif key == 'all':
            samples = self.samples
        else:
            raise ValueError("Invalid key, key must be either 'all', 'test', or 'train")
        
        return self.__Qnet_formatter(samples, key=key)
    
    def __set_varcase(self,
                      lower,
                      key='train',
                      vars=None):
        """set the features to all upper or lowercase

        Args:
          lower (bool): If true, set vars to lowercase, else to uppercase
          key (str, optional): Whether to set train or test features. Defaults to 'train'.
          vars ([str]): Mutable and immutable vars/features. Defaults to None.

        Returns:
          features, vars: formatted to either upper or lower case
        """
        if lower:
            features = [x.lower() for x in self.features[key]]
            if var is not None:
                vars = [x.lower() for x in vars]
        else:
            features = [x.upper() for x in self.features[key]]
            if vars is not None:
                vars = [x.upper() for x in vars]
        return features, vars

    def __interpretvars(self,
                        lower,
                        IMMUTABLE,
                        FILE=None,
                        LIST=None):
        """read in vars from file and set mutable, immutable

        Args:
          lower (bool): Whether to set variables to lowercase (True) or not (False)
          IMMUTABLE (book): IMMUTABLE if True, MUTABLE otherwise
          FILE (str, optional): file with vars in singular column. Defaults to None.
          LIST ([str], optional): 1D array of vars. Defaults to None.
          
        Returns:
          mutable vars, immutable vars: list
        """
        if IMMUTABLE:
            immutable_vars = np.array(LIST)
            if FILE is not None:
                immutable_vars = pd.read_csv(FILE,index_col=0).transpose()
            #assert_array_dimension(immutable_vars, 1)
            features, immutable_vars = self.__set_varcase(immutable_vars,
                                                          lower)
            mutable_vars = [x for x in features
                            if x not in immutable_vars]
            immutable_vars = [x for x in immutable_vars
                              if x in features]
            invalid_vars = [x for x in immutable_vars
                            if x not in features]
        else:
            mutable_vars = LIST
            if FILE is not None:
                mutable_vars = pd.read_csv(FILE,index_col=0).transpose()
            #assert_array_dimension(mutable_vars, 1)
            features, mutable_vars = self.__set_varcase(mutable_vars,
                                                        lower)
            immutable_vars = [x for x in features
                              if x not in mutable_vars]
            mutable_vars = [x for x in mutable_vars
                            if x in features]
            invalid_vars = [x for x in mutable_vars
                            if x not in features]
        if len(invalid_vars) != 0:
            print("{} vars not found".format(len(invalid_vars)))
            print("vars not found:{}".format(invalid_vars))
        return mutable_vars, immutable_vars

    def mutable_variables(self,
                immutable_list=None,
                IMMUTABLE_FILE=None,
                mutable_list=None,
                MUTABLE_FILE=None,
                lower=False):
        """set variables to be mutable or immutable

        Args:
          immutable_list (list)): 1D array of immutable variables. Defaults to None.
          IMMUTABLE_FILE (str, optional): file with immutable vars in singular column. Defaults to None.
          mutable_list (list, optional): 1D array of immutable variables. Defaults to None.
          MUTABLE_FILE (str, optional): file with mutable vars in singular column. Defaults to None.
          
        Returns:
          mutable_vars, immutable_vars: list
        """
        list_None = assert_None([immutable_list,mutable_list], raise_error=False)
        file_None = assert_None([IMMUTABLE_FILE,MUTABLE_FILE], raise_error=False)
        num_None = assert_None([immutable_list,mutable_list,
                                IMMUTABLE_FILE,MUTABLE_FILE], raise_error=False)
        if list_None == 0 or file_None == 0:
            raise ValueError("Only input either IMMUTABLE or MUTABLE vars, not both!")
        elif num_None == 4:
            raise ValueError("Too few inputs! One argument needed")
        elif num_None != 3:
            raise ValueError("Too many inputs! Only one argument needed")
        else:
            if IMMUTABLE_FILE is not None:
                mutable_vars, immutable_vars = self.__interpretvars_fromfile(IMMUTABLE=True,
                                                                             FILE=IMMUTABLE_FILE,
                                                                             lower=lower)
            elif MUTABLE_FILE is not None:
                mutable_vars, immutable_vars = self.__interpretvars_fromfile(IMMUTABLE=False,
                                                                             FILE=MUTABLE_FILE,
                                                                             lower=lower)
            elif immutable_list is not None:
                mutable_vars, immutable_vars = self.__interpretvars_fromfile(IMMUTABLE=True,
                                                                             LIST=immutable_list,
                                                                             lower=lower)
            elif mutable_list is not None:
                mutable_vars, immutable_vars = self.__interpretvars_fromfile(IMMUTABLE=False,
                                                                             LIST=mutable_list,
                                                                             lower=lower)
        self.mutable_vars, self.immutable_vars = mutable_vars, immutable_vars
        return mutable_vars, immutable_vars            