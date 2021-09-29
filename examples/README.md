# Calculating distance matrix with mpi using cognet class
## 1. Initialize cognet class, either from model and dataformatter objects, or directly inputting data:
<pre><code>from cognet.cognet import cognet as cg
from cognet.model import model
from cognet.dataFormatter import dataFormatter
import pandas as pd
import numpy as np

data_ = dataFormatter(samples='examples_data/gss_2018.csv')
model_ = model()
model_.load("examples_data/gss_2018.joblib")
cognet_ = cg()
cognet_.load_from_model(model_, data_, 'all')
</code></pre>

## 2. For smaller and less intensive datasets, use cognet.distfunc_multiples:
<pre><code>distance_matrix=cognet_.distfunc_multiples("examples_results/distfunc_multiples_testing.csv")</code></pre>
  
## 3. For larger and more intensive datasets, first call cognet.dmat_filewriter to write the necessary files:
<pre><code>Cg.dmat_filewriter("GSS_cognet.py", "examples_data/gss_2018.joblib",
                    MPI_SETUP_FILE="GSS_mpi_setup.sh",
                    MPI_RUN_FILE="GSS_mpi_run.sh",
                    MPI_LAUNCHER_FILE="GSS_mpi_launcher.sh",
                    YEARS='2018',NODES=4,T=14)
</code></pre>

## 4. Make any changes necessary to the run and setup scripts and pyfile, then call the run script in the terminal:
<pre><code>from subprocess import call
call(["./GSS_mpi_run.sh"])
</code></pre>
