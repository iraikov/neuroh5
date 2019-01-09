#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=1024:ppn=16:xe
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
#PBS -A baqc

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

results_path=./results/Full_Scale_Control_$PBS_JOBID
export results_path
export SCRATCH=/projects/sciteam/baqc

export MPICH_RANK_REORDER_METHOD=2
export MPICH_ALLTOALLV_THROTTLE=2


mkdir -p $results_path

aprun -n 16384 ./build/neurograph_scatter  -a "Synapse Attributes" -i 128 \
      $SCRATCH/Full_Scale_Control/DG_Connections_Full_Scale_20181017.h5



