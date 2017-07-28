#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=4:00:00
### set the job name
#PBS -N comet_rsync
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

aprun -n 1 rsync -avz \
    /projects/sciteam/baef/Full_Scale_Control/dentate_Full_Scale_Control_MPP_20170313.h5 \
    iraikov@comet.sdsc.edu:/oasis/scratch/comet/iraikov/temp_project

