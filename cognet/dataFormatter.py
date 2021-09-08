import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from cognet.util import assert_None, assert_array_dimension
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
        self.samples = pd.read_csv(samples)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = None
        self.train_data, self.test_data = train_test_split(self.samples,
                                                           test_size=test_size,
                                                           train_size=train_size,
                                                           random_state=random_state)
        self.features = {}
        self.nan_cols = []
        self.immutable_vars = None
        self.mutable_vars = None

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
        # if not isinstance(samples, np.ndarray):
        #     raise ValueError('Samples must be in numpy array form!')
        samples = samples
        features = np.array(samples.columns.astype(str)[:])
        samples = samples.values.astype(str)[:]
        # remove columns that are all NaNs
        not_all_nan_cols = ~np.all(samples == '', axis=0)
        self.nan_cols = np.all(samples == '', axis=0)

        samples = samples[:, not_all_nan_cols]
        
        features = features[not_all_nan_cols]
        features = list(features)
        self.features[key] = features
        return features, samples

    def train(self):
        """return train data
        """
        return self.__Qnet_formatter('train',self.train_data)
    
    def test(self):
        """return test data
        """
        return self.__Qnet_formatter('test',self.test_data)
    
    def __set_varcase(self,
                      vars,
                      lower):
        """[summary]

        Args:
            vars ([type]): [description]
            lower ([type]): [description]

        Returns:
            [type]: [description]
        """
        if lower:
            features = [x.lower() for x in self.features['train']]
            vars = [x.lower() for x in vars]
        else:
            features = [x.upper() for x in self.features['train']]
            vars = [x.upper() for x in vars]
        return features, vars

    def __interpretvars_fromfile(self,
                                 lower,
                                 IMMUTABLE,
                                 FILE=None,
                                 LIST=None):
        """[summary]

        Args:
            IMMUTABLE ([type]): [description]
            FILE ([type]): [description]
            lower ([type]): [description]

        Returns:
            [type]: [description]
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
        ## can set arguments to accept any type,
        ## and add parameters to make sure if list or FILE, immutable or mutable
        """[summary]

        Args:
            immutable_list ([type]): [description]
            IMMUTABLE_FILE (str, optional): [description]. Defaults to ''.
            mutable_list (list, optional): [description]. Defaults to [].
            MUTABLE_FILE (str, optional): [description]. Defaults to ''.

        Raises:
            ValueError: [description]
            ValueError: [description]
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
