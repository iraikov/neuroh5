#!/bin/bash
#
#SBATCH -J test_read_coords
#SBATCH -o ./results/test_read_coords.%j.o
#SBATCH -n 64
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python/2.7.5
module load mpich/3.1.4/gcc
module load gcc/4.9.1

export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export PYTHONPATH=$HOME/bin/nrn/lib64/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/bin/hdf5/lib:$LD_LIBRARY_PATH

set -x

mpirun -np 16 python ./tests/test_read_coords.py \
       --coords-path=$SCRATCH/dentate/dentate_Full_Scale_Control_coords_20171005.h5 \
       --coords-namespace=Coordinates \
       --io-size=1

