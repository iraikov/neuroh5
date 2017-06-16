#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=128:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N synimport
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
connectivity_output_path=$prefix/dentate_Full_Scale_GC_20170510.h5


aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
      -d $forest_connectivity_path:/Populations/GC/Connectivity \
      MPP GC MPPtoGC \
      $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
      -d $forest_connectivity_path:/Populations/GC/Connectivity \
      LPP GC LPPtoGC \
      $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       MC GC MCtoGC \
       $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       BC GC BCtoGC \
       $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       HC GC HCtoGC \
       $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       AAC GC AACtoGC \
       $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       HCC GC HCCtoGC \
       $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       NGFC GC NGFCtoGC \
       $connectivity_output_path

aprun -n 2048 ./build/neurograph_import -f hdf5:syn -s 256 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       MOPP GC MOPPtoGC \
       $connectivity_output_path
