#!/bin/bash
#PBS -l nodes=64:ppn=16:xe
#PBS -q debug
#PBS -l walltime=0:30:00
#PBS -e ./results/test_cell_attr_gen.$PBS_JOBID.err
#PBS -o ./results/test_cell_attr_gen.$PBS_JOBID.out
#PBS -N test_cell_attr_gen
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A baqc


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

set -x
cd $PBS_O_WORKDIR

aprun -n 1024 -b -- bwpy-environ -- python2.7 ./tests/test_cell_attr_gen.py 

