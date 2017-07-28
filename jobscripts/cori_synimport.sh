#!/bin/bash
#
#SBATCH -J synimport
#SBATCH -o ./results/synimport.%j.o
#SBATCH -N 32
#SBATCH --ntasks-per-node=32
#SBATCH -p default
#SBATCH -t 0:30:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module swap PrgEnv-intel PrgEnv-gnu
module unload darshan
module load cray-hdf5-parallel/1.8.16

set -x


prefix=$SCRATCH/dentate/Full_Scale_Control/
export prefix

forest_connectivity_path=$prefix/DGC_forest_connectivity_20170508.h5
connectivity_output_path=$prefix/dentate_Full_Scale_GC_AAC_20170727.h5

srun -n 1 python ./tests/test_networkit.py

mpirun -np 2048 ./build/neurograph_import -f hdf5:syn -s 128 \
       -d $forest_connectivity_path:/Populations/GC/Connectivity \
       AAC GC $connectivity_output_path
