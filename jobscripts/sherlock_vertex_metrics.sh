#!/bin/bash
#
#SBATCH -J vertex_metrics_DG_IN
#SBATCH -o ./results/vertex_metrics_DG_IN.%j.o
#SBATCH -n 128
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python/2.7.5
module load mpich/3.1.4/gcc
module load gcc/4.9.1

export SCRATCH=/scratch/users/iraikov
export LD_LIBRARY_PATH=$HOME/bin/hdf5/lib:$LD_LIBRARY_PATH

set -x


mpirun -np 128 ./build/neurograph_vertex_metrics  $SCRATCH/dentate/DG_IN_connections_20171014.h5 -i 64

