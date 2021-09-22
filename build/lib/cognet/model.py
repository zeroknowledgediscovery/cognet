from quasinet.qnet import Qnet, load_qnet, save_qnet
from quasinet.qnet import export_qnet_tree, export_qnet_graph
from cognet.util import assert_None
class model:
    """Facilitate training and constructing Qnet
    """

    def __init__(self):
        """Init
        """
        self.myQnet = None
        self.features = None
        self.immutable_vars = None
        self.mutable_vars = None
        self.data_obj = None

    def fit(self,
            featurenames=None,
            samples=None,
            data_obj=None,
            njobs=4):
        """fit Quasinet Qnet model

        Args:
          featurenames ([str], optional): names of the model features. Defaults to None.
          samples ([str], optional): 2D array with rows as observations and columns as features. Defaults to None.
          data_obj (obj, optional): Build Qnet directly from data obj without other inputs. Defaults to None.
          njobs (int, optional): Number of jobs used to fit Qnet. Defaults to 2.
        """
        num_None = assert_None([featurenames,samples,data_obj], raise_error=False)
        if num_None == 0:
            raise ValueError("input either samples and features or data object, not both!")
        elif data_obj is not None:
            featurenames, samples=data_obj.train() # returns the training data
            self.immutable_vars, self.mutable_vars = data_obj.immutable_vars, data_obj.mutable_vars
        elif num_None > 1:
            raise ValueError("input both samples and features or data object!")
        self.myQnet = Qnet(n_jobs=njobs, feature_names=featurenames)
        self.myQnet.fit(samples)
        self.features = featurenames

    def save(self,
             file_path=None):
        """save qnet

        Args:
          file_path (str, optional): Desired Qnet filename. Defaults to None.
        """
        assert_None([self.myQnet])
        if file_path is None:
            file_path = 'tmp_Qnet.joblib'
        save_qnet(self.myQnet, file_path)
    
    def load(self,
             file_path):
        """load Qnet from file

        Args:
          file_path (str): path to Qnet savefile

        Returns:
          [Qnet]: Qnet object
        """
        print("updating")
        self.myQnet = load_qnet(file_path)
        self.features = self.myQnet.feature_names
        return self.myQnet

    def export_dot(self,
                   filename,
                   index=[3],
                   path='',
                   generate_trees=False,
                   threshold=0.2):
        """export Qnet trees

        Args:
          filename (str): Desired tree savefile
          index (list, optional): list of indices to generate trees. Defaults to [3].
          path (str, optional): Desired tree savefile path. Defaults to ''.
          generate_trees (bool, optional): Whether or not to generate individual trees. 
                                             Defaults to False, or to generate individual trees.
          threshold (float, optional): Numeric cutoff for edge weights. If the edge weights exceed 
                                         this cutoff, then we include it into the graph. Defaults to 0.2.
        """
        if not generate_trees:
            export_qnet_graph(self.myQnet, 
                              threshold, path+filename)
        else:
            for i in index:
                export_qnet_tree(self.myQnet,
                                 i, filename)
