#!/bin/bash
#
#SBATCH -J synimport
#SBATCH -o ./results/synimport.%j.o
#SBATCH -N 32
#SBATCH --ntasks-per-node=32
#SBATCH -p regular
#SBATCH -t 2:00:00
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
connectivity_output_path=$prefix/dentate_Full_Scale_GC_20170728.h5

for post in GC; do

for pre in MPP LPP MC BC HC AAC HCC NGFC MOPP; do

srun -n 1024 ./build/neurograph_import -f hdf5:syn -s 128 \
      -d $forest_connectivity_path:/Populations/$post/Connectivity \
      $pre $post $connectivity_output_path

done
done
