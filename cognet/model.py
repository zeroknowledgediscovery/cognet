from quasinet.qnet import Qnet, load_qnet, save_qnet
from quasinet.qnet import export_qnet_tree, export_qnet_graph

class model:
    """Facilitate training and constructing Qnet
    """

    def __init__(self):
        """Init
        """
        self.myQnet = None

    def fit(self,
            featurenames,
            samples=None,
            data_obj=None,
            njobs=2):
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
        num_None = assert_None([samples,data_obj], False)

        if num_None == 1:
            self.myQnet = Qnet(n_jobs=njobs, feature_names=featurenames)
            if data_obj is not None:
                samples=data_obj.train() # returns the training data
            myQnet.fit(samples)

        elif num_None == 2:
            raise ValueError("input either samples or data object!")
        elif num_None == 0:
            raise ValueError("input either samples or data object, not both!")

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
        self.myQnet = load_qnet(file_path)
        return self.myQnet

    def export_dot(self,
                   filename,
                   index=[6],
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
                                 i, path_filename)