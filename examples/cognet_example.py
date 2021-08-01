# This is an example of how we see 
# the package work. The functions listed here
# are probably teh only ones that should be exposed, ie documented.
# others should br prepended with a double underscore
#  
# The cognet directory has the following "modules"
# which are seprate .py files containing clases and functions
# The modules are cognet.py, dataFormatter.py, model.py, util.py, viz.py
# we will write the viz.py later.
​
from cognet.cognet import cognet
from cognet.dataFormatter import dataFormatter
from cognet.model import model 
from cognet.util import embed
from cognet.viz import distance_contour 
​
data = dataFormatter(samples=samples,
                     test_size=0.5)
    # load the sample data
    # have option for test/train split
    # make checks to ensure we will not bark at qnet construction 
    # data.train() returns traininh data
    # data.test() returns test data
    

# we can set mutable and immutable vars from list or file
mutable_vars, immutable_vars = data.set_vars(immutable_list=list)
mutable_vars, immutable_vars = data.set_vars(MUTABLE_FILE=file)
features,samples = data.train()
​
# can either input features and samples directly, or infer from data obj
model_ = model()
model_.fit(data_obj=data)
        # qnet construction parameters 
        # infer qnet
        
model_.export_dot(dotsavepath,
                 dotfile_prefix,
                 generate_trees=True)
model_.save(savepath)
model_.load(savepath)
​
# set some paramaters in instantiating cognet class 
# also setup Dnull, and nullbaseFreq
# if loading from model obj, no need to load_data, otherwise, load_data
Cg=cognet(model,samples, **kwargs)
    
# distance calculation for individual samples    
# we have a nsteps parameter (for sample 1 and sample2)
# which qsamples the sample1 and sample2 if set before
# computing distance. Note qsampling must only 
# change mutable varaibles, so need to compute base-freq
distance = Cg.distance(sample1,sample2,nsteps=[None,None])
distance = Cg.distance(data.samples[0],data.samples[1])
distance = Cg.distance(data.test[0],data.test[1])
​
# produce stats on how many column names actually match
stast = Cg.set_poles(POLEFILE,steps)
​
# compute polar distance matrix
dmatrix = Cg.polarSeparation(nsteps=0)
​
# the following are for single samples
#------------------
dissonance_array = Cg.dissonance(sample)
dataframes,error_array = Cg.randomMaskReconstruction(sample)
ideology_index = Cg.ideology(sample)
local_dispersion = Cg.dispersion(sample)
# compute distance from each pole
array_distances = Cg.polarDistance(sample)
#-------------------
​
#the following must use parallelization
# next one must use mpi and hence will not run
# with mpi without maybe a seprate script.
# But look here: https://stackoverflow.com/questions/25772289/python-multiprocessing-within-mpi
ditance_matrix=Cg.distance_matrix(samples,nsteps=0)
​
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