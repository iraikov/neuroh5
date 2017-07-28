#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=128:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N synimport_test
### set the job stdout and stderr
#PBS -e ./results/synimport.$PBS_JOBID.err
#PBS -o ./results/synimport.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

prefix=/projects/sciteam/baef/Full_Scale_Control
export prefix

forest_connectivity_path=$prefix/DGC_forest_connectivity_20170508.h5
connectivity_output_path=$prefix/dentate_Full_Scale_GC_AAC_20170727.h5

for post in GC; do

for pre in AAC; do

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 128 \
      -d $forest_connectivity_path:/Populations/$post/Connectivity \
      $pre $post $connectivity_output_path

done
done
