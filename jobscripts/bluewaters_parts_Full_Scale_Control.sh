#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=24:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N parts_Full_Scale_Control
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

#module load cray-tpsl
module load cray-hdf5-parallel

set -x

export LD_LIBRARY_PATH=$HOME/bin/parmetis/lib:$LD_LIBRARY_PATH

cd $PBS_O_WORKDIR

results_path=./results/Full_Scale_Control_$PBS_JOBID
export results_path

mkdir -p $results_path

aprun -n 256 ./parts/src/parts \
      /u/sciteam/raikov/scratch/dentate/dentate_Full_Scale_Control_PP.h5 \
      -i 64 -n 4096 -o ${results_path}/parts



