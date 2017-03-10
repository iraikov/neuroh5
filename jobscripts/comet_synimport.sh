#!/bin/bash
#
#SBATCH -J neurograph_synimport
#SBATCH -o ./results/neurograph_synimport.%j.o
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x

mpirun -np 128 ./build/neurograph_import -f hdf5:syn --src-offset=-44990 \
      -d /oasis/scratch/comet/iraikov/temp_project/DGC_forest_connectivity_test.h5:/Populations/GC/Connectivity \
      MPP GC MPPtoGC \
      /oasis/scratch/comet/iraikov/temp_project/dentate_Reduced_Scale_Control_MPP.h5 
