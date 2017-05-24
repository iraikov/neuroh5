#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:ppn=8:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N read_neurotrees_py
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

export PYTHONPATH=/projects/sciteam/baef/site-packages:$PYTHONPATH

set -x

aprun -n 8 python ./tests/reader_test.py
