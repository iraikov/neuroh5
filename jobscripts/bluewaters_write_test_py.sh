#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=2:ppn=8:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N write_cell_attrs_test_py
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load bwpy
module load bwpy-mpi
module load cray-hdf5-parallel

cd $PBS_O_WORKDIR

export PYTHONPATH=/projects/sciteam/baqc/site-packages:$PYTHONPATH

set -x

cp ./data/example_h5types.h5 ./data/write_cell_attr.h5 && \
aprun -n 16 -b -- bwpy-environ -- \
python2.7 ./tests/test_write_cell_attr.py
