import os
import sys
import subprocess as sp

run = ''

py2 = sys.version_info[0] < 3

run_file = sys.argv[1]
dataset = sys.argv[2]
assert ".py" in run_file

if py2:
    #change python version if needed
    py_load = 'python/3.7.0'
    py_ver='python3'
else:
    py_load = 'python/3.7.0' #'python/intel-2020.up1' load version of python
    py_ver='python'
    
print('Running with MPI ')
py_ver='mpirun python3 -m mpi4py.futures'
run = run + "_MPI"
t_per_node = 28

first ="""#!/bin/bash 
#SBATCH --job-name={args:.4}{run:.4}{dataset:.4}
#SBATCH --output=./examples_results/{args:.4}{run:.4}{dataset:.4}.out
#SBATCH --error=./examples_results/{args:.4}{run:.4}{dataset:.4}.err
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node={t_per_node}
#SBATCH --partition=broadwl
date; 
module load midway2
module unload python
module load python/anaconda-2020.02
module unload python
module load python/cpython-3.7.0
{py_ver} {args};
date; """.format(dataset=dataset,py_load=py_load, py_ver=py_ver, run=run, t_per_node=t_per_node, args=' '.join(sys.argv[1:]))


name_file ="examples_results/quick_runab.sbc"
with open(name_file, 'w') as f:
    f.write(first)   
sp.call('sbatch '+name_file,shell=True)
sp.call('rm '+name_file,shell=True)
#format: python submit_one.py my_code.py job_name