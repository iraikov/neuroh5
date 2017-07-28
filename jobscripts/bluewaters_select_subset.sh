#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=64:ppn=16:xe
### which queue/account to use
#PBS -q debug
#PBS -A bafv
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N select_neurotrees
### set the job stdout and stderr
#PBS -e ./results/neurotrees_select.$PBS_JOBID.err
#PBS -o ./results/neurotrees_select.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

aprun -n 1024 ./build/neurotrees_select -p GC -i 256 \
      --cachesize=$((4 * 1024 * 1024)) \
      /projects/sciteam/baef/Full_Scale_Control/DGC_forest_20170711.h5 \
      /projects/sciteam/baef/Full_Scale_Control/DGC_forest_selection_20170714.dat \
      /projects/sciteam/baef/Full_Scale_Control/DGC_forest_subset_100_20170714.h5 




