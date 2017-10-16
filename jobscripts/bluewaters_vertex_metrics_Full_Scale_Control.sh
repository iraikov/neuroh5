#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=32:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N vertex_metrics_GC
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

export input_file="/projects/sciteam/baef/Full_Scale_Control/dentate_Full_Scale_GC_20170902.h5"

aprun -n 512 ./build/vertex_metrics -i 128 $input_file



