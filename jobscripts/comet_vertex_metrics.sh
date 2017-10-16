#!/bin/bash
#
#SBATCH -J neurograph_vertex_metrics
#SBATCH -o ./results/neurograph_vertex_metrics.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x

ibrun -np 1024 ./build/neurograph_vertex_metrics \
      $SCRATCH/dentate/DG_GC_connections_20171013_compressed.h5 \
      -i 256

echo All done!


