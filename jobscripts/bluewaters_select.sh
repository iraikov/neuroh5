#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=64:ppn=16:xe
### which queue/account to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=1:00:00
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

aprun -n 1024 ./build/neurotrees_select -p GC -i 256 --reindex \
      --cachesize=$((4 * 1024 * 1024)) \
      /projects/sciteam/baef/Full_Scale_Control/DGC_forest_extended_20171019_compressed.h5 \
      /projects/sciteam/baef/Full_Scale_Control/DGC_forest_reindex_20170615.dat \
      /projects/sciteam/baef/Full_Scale_Control/DGC_forest_20171019.h5 




