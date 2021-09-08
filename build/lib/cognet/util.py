from sklearn.decomposition import PCA
import pandas as pd

def assert_None(args,
                raise_error=True):
    '''Make sure args are not None
    '''
    if any(x is None for x in args):
        num_none = sum(x is None for x in args)
        if raise_error:
            string='Nones detected : {}'.format(str(num_none))
            raise ValueError(string)
        else:
            return num_none

def assert_array_dimension(array, dimensions):
    '''
    '''
    if len(array.shape) != dimensions:
        raise ValueError('You must pass in a {}-D array!'.format(dimensions))
    
def embed_to_pca(EFILE, OUTFILE):
    """[summary]

    Args:
        EFILE ([type]): [description]
        OUTFILE ([type]): [description]
    """
    Ef=pd.read_csv(EFILE,sep=' ',header=None).dropna(axis=1).transpose()
    Ef.columns=['x'+str(i) for i in Ef.columns]
    xf=Ef#.assign(IF=dx.ido)

    pca = PCA(n_components=2).fit(xf)
    ef=pca.fit_transform(xf)
    pd.DataFrame(ef).to_csv(OUTFILE,header=None,index=None)
    return ef
        