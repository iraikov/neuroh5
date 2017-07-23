#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=256:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:10:00
### set the job name
#PBS -N scatter_Full_Scale_Control_4096
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
### Get darsan profile data
#PBS -lgres=darshan

module load cray-hdf5-parallel/1.8.16
module load gcc/4.9.3

set -x

cd $PBS_O_WORKDIR

export LD_LIBRARY_PATH=$HOME/bin/parmetis/lib:$LD_LIBRARY_PATH

results_path=./results/Full_Scale_Control_4096_$PBS_JOBID
export results_path

aprun -n 4096 ./build/scatter -a -i 256 \
      /projects/sciteam/baef/Full_Scale_Control/dentate_Full_Scale_Control_20170510.h5


