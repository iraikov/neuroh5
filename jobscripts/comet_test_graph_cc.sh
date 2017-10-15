#!/bin/bash
#
#SBATCH -J neurograph_test_graph_cc
#SBATCH -o ./results/neurograph_test_graph_cc.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#
module load intel/2015.2.164
module load hdf5
module load python
module load mpi4py

export PYTHONPATH=/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

set -x

PYTHONPATH=$PYTHONPATH \
mpirun -np 1024 python ./tests/test_graph_cc.py
 ./build/vertex_metrics \
      /u/sciteam/raikov/scratch/dentate/dentate_Full_Scale_Control_PP.h5 \
      -i 64

echo All done!


