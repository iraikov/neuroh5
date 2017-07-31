#!/bin/bash
#
#SBATCH -J test_networkit
#SBATCH -o ./results/test_networkit.%j.o
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p debug
#SBATCH -t 0:30:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module unload darshan
module load cray-hdf5-parallel/1.8.16
module load python/3.5-anaconda

set -x

srun -n 1 python ./tests/test_networkit.py
