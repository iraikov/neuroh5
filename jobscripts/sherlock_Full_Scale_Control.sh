#!/bin/bash
#
#SBATCH -J summary_dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH -n 32
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load mpich/3.1.4/gcc

export SCRATCH=/scratch/users/iraikov
export LD_LIBRARY_PATH=$PI_HOME/hdf5:$LD_LIBRARY_PATH

set -x


mpirun ./build/reader -s $SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170728.h5

