#!/bin/bash
#
#SBATCH -J neurograph_scatter_attrs_test
#SBATCH -o ./results/neurograph_scatter_attrs_test.%j.o
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load hdf5

ibrun ./build/scatter -a -i 32 \
 -o /oasis/scratch/comet/iraikov/temp_project/dentate_Reduced_Scale_Control \
 /oasis/scratch/comet/iraikov/temp_project/dentate_Reduced_Scale_Control_MPP.h5


