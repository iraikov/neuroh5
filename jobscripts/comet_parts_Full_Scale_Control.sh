#!/bin/bash
#
#SBATCH -J neurograph_parts
#SBATCH -o ./results/neurograph_parts.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load hdf5

ibrun ./build/balance_indegree  -i 128 -n 1024 -o ./results/parts.1024 \
 /oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_Control_MPP.h5


