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
        self.samples = None
        self.immutable_vars = None
        self.mutable_vars = None

    def fit(self,
            featurenames=None,
            samples=None,
            data_obj=None,
            njobs=4):
        """[summary]

        Args:
          featurenames ([type]): [description]
          samples ([type], optional): [description]. Defaults to None.
          data_obj ([type], optional): [description]. Defaults to None.
          njobs (int, optional): [description]. Defaults to 2.

        Raises:
            ValueError: [description]
            ValueError: [description]
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
        print("qnet")
        self.myQnet.fit(samples)
        print("done")
        self.samples = samples
        self.features = featurenames

    def save(self,
             file_path=None):
        """[summary]

        Args:
          file_path ([type], optional): [description]. Defaults to None.
        """
        assert_None([self.myQnet])
        if file_path is None:
            file_path = 'tmp_Qnet.joblib'
        save_qnet(self.myQnet, file_path)
    
    def load(self,
             file_path):
        """[summary]

        Args:
          file_path ([type]): [description]

        Returns:
          [type]: [description]
        """
        ## can also directly use load from joblib, thoughts?
        print("updating")
        self.myQnet = load_qnet(file_path)
        self.features = self.myQnet.feature_names
        return self.myQnet

    def export_dot(self,
                   filename,
                   index=[3],
                   path='',
                   generate_trees=False,
                   threhold=0.2):
        """[summary]

        Args:
            filename ([type]): [description]
            index (list, optional): [description]. Defaults to [6].
            path (str, optional): [description]. Defaults to ''.
            generate_trees (bool, optional): [description]. Defaults to False.
            threhold (float, optional): [description]. Defaults to 0.2.
        """
        if not generate_trees:
            export_qnet_graph(self.myQnet, 
                              threshold, path+filename)
        else:
            for i in index:
                export_qnet_tree(self.myQnet,
                                 i, filename)
