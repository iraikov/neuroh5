#!/bin/bash
#
#SBATCH -J neurograph_scatter_attrs_test
#SBATCH -o ./results/neurograph_scatter_attrs_test.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5
module load python
module load mpi4py

export LD_PRELOAD=$MPIHOME/lib/libmpi.so
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

set -x

ibrun -np 512 python ./tests/comet_scatter_test.py

echo All done!


