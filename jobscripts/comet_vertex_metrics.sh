#!/bin/bash
#
#SBATCH -J neurograph_vertex_metrics
#SBATCH -o ./results/neurograph_vertex_metrics.%j.o
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x

ulimit -c unlimited

SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export SCRATCH

##      $SCRATCH/dentate/DG_GC_connections_20171013_compressed.h5

ibrun -np 320 ./build/neurograph_vertex_metrics --outdegree \
      $SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20171110.h5 \
      -i 192

echo All done!


