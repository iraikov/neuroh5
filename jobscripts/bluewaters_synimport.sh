#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=8:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N synimport
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module load cray-hdf5-parallel
module load gcc/4.9.3

set -x

cd $PBS_O_WORKDIR

aprun -n 128 ./build/neurograph_import --src-offset=-44990 \
      MPP GC MPPtoGC \
      $HOME/scratch/dentate/dentate_Full_Scale_Control_MPP.h5 \
      -f /projects/sciteam/baef/DGC_forest_connectivity.h5:/Populations/GC/Connectivity
