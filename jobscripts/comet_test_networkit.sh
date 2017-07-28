#!/bin/bash
#
#SBATCH -J neurograph_test_networkit
#SBATCH -o ./results/neurograph_test_networkit.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute
#SBATCH -t 0:10:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#
module load python
module load hdf5


set -x

mpirun -np 1 python3 ./tests/test_networkit.py

echo All done!


