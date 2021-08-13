# This is an example of how we see 
# the package work. The functions listed here
# are probably teh only ones that should be exposed, ie documented.
# others should br prepended with a double underscore
#  
# The cognet directory has the following "modules"
# which are seprate .py files containing clases and functions
# The modules are cognet.py, dataFormatter.py, model.py, util.py, viz.py
# we will write the viz.py later.
import sys
sys.path.append("../")

from cognet.cognet import cognet
print(cognet)
from cognet.dataFormatter import dataFormatter
from cognet.model import model 
#import cognet.util
import pandas as pd
#from cognet.viz import distance_contour 

yr = '2018'
GSS_dir = '../../creed2_/GSS/data/processed_data/'
POLEFILE='../../creed2_/GSS/polar_vectors.csv'
GSS_FEATURES_BY_YEAR='../../creed2_/GSS/data/features/features_by_year_GSS.csv'
GSSDATA=GSS_dir+'gss_'+yr+'.csv'
QPATH='../../creed2_/GSS/Qnets/gss_'+yr+'.joblib'
IMMUTABLE_FILE='../../creed2_/GSS/immutable.csv'

test_dataFormatter = True
test_model = False
if test_dataFormatter:
        data = dataFormatter(samples=GSSDATA,
                        test_size=0.5)
        # load the sample data
        # have option for test/train split
        # make checks to ensure we will not bark at qnet construction 
        # data.train() returns traininh data
        # data.test() returns test data
        
        features,samples = data.train()
        features,samples = data.train()
        # we can set mutable and immutable vars from list or file
        im_vars_df = pd.read_csv(IMMUTABLE_FILE, names=['vars'])
        im_vars_list = im_vars_df.vars.to_list()
        mutable_vars, immutable_vars = data.set_vars(immutable_list=im_vars_list)
        mutable_vars, immutable_vars = data.set_vars(IMMUTABLE_FILE=IMMUTABLE_FILE)


        testing = False
        # can either input features and samples directly, or infer from data obj
        
        model_ = model()
        model_.fit(data_obj=data)
        # qnet construction parameters 
        # infer qnet
        
        if test_model:
                model_.export_dot("tmp_dot_modelclass.dot",
                                generate_trees=True)
                model_.save("tmp_nodelclass.joblib")
                model_.load("tmp_nodelclass.joblib")
        
        # set some paramaters in instantiating cognet class 
        # also setup Dnull, and nullbaseFreq
        # if loading from model obj, no need to load_data, otherwise, load_data
         
        Cg = cognet()
        Cg.load_from_model(model_)
        
        # distance calculation for individual samples    
        # we have a nsteps parameter (for sample 1 and sample2)
        # which qsamples the sample1 and sample2 if set before
        # computing distance. Note qsampling must only 
        # change mutable varaibles, so need to compute base-freq
        distance = Cg.distance(samples[1],samples[3],nsteps1=5, nsteps2=5)
        #distance = Cg.distance(data.samples[0],data.samples[1])
        #distance = Cg.distance(data.test[0],data.test[1])

        # produce stats on how many column names actually match
        stats = Cg.set_poles(POLEFILE,steps)

        # compute polar distance matrix
        dmatrix = Cg.polar_separation(nsteps=0)

        # the following are for single samples
        #------------------
        dissonance_array = Cg.dissonance(sample)
        rederr,r_prob,rand_err = Cg.randomMaskReconstruction(sample)
        #ideology_index = Cg.compute_DLI_sample(3)
        local_dispersion = Cg.compute_DLI_sample(3)[4:]
        # compute distance from each pole
        array_distances = Cg.polarDistance(sample)
        #-------------------

        #the following must use parallelization
        # next one must use mpi and hence will not run
        # with mpi without maybe a seprate script.
        # But look here: https://stackoverflow.com/questions/25772289/python-multiprocessing-within-mpi
        distance_matrix=Cg.qdistance_matrix(samples,nsteps=0)

        # multiprocessing suffices
        dissonance_array = Cg.dissonance_multiple(samples)
        # multiprocessing suffices
        dataframes,error_array = Cg.randomMaskReconstruction_multiple(samples)
        # multiprocessing suffices
        ideology_index = Cg.ideology_multiple(samples)
        # multiprocessing suffices
        local_dispersion = Cg.dispersion_multiple(samples)
        # compute distance from each pole
        # multiprocessing suffices
        array_distances = Cg.polarDistance_multiple(samples)
        #-------------------