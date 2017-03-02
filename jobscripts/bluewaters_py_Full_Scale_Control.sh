#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=256:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N scatter_py_Full_Scale_Control
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
### Get darsan profile data
#PBS -lgres=darshan

module load cray-hdf5-parallel
module load bwpy bwpy-mpi

set -x

cd $PBS_O_WORKDIR


aprun -n 4096 python ./tests/bw_scatter_test.py




