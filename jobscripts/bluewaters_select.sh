#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=32:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N select_neurotrees
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

rm -f /projects/sciteam/baef/DGC_forest_subset_1000.h5 

aprun -n 512 ./neurotrees_select -a -n Synapse_Attributes -p GC -i 128  \
      --cachesize=$((4 * 1024 * 1024)) \
      /projects/sciteam/baef/DGC_forest_syns_bak.h5 \
      /projects/sciteam/baef/DGC_forest_subset_selection.dat \
      /projects/sciteam/baef/DGC_forest_subset_1000.h5 




