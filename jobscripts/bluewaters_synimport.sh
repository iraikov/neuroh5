#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=16:ppn=16:xe
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

set -x

cd $PBS_O_WORKDIR

aprun -n 256 ./build/neurograph_import -f hdf5:syn --src-offset=-44990 -s 256 \
      -d /projects/sciteam/baef/DGC_forest_syn_connectivity_20170313.h5:/Populations/GC/Connectivity \
      MPP GC MPPtoGC \
      /projects/sciteam/baef/dentate_Full_Scale_Control_MPP_20170313.h5 
