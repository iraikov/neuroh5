#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=32:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N scatter_Full_Scale_Control
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
module load gcc/4.9.3

set -x

cd $PBS_O_WORKDIR

results_path=./results/Full_Scale_Control_$PBS_JOBID
export results_path

mkdir -p $results_path

aprun -n 512 ./reader/src/scatter -a -i 64  \
      /u/sciteam/raikov/scratch/dentate/dentate_Full_Scale_Control_dbs.h5



