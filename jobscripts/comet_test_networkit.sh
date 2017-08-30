#!/bin/bash
#
#SBATCH -J neurograph_test_networkit
#SBATCH -o ./results/neurograph_test_networkit.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=455G
#SBATCH -p large-shared
#SBATCH -t 18:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#
module unload intel
module load gnu
module load openmpi_ib
module load python
module load hdf5

set -x

srun --mpi=pmi2 -n 1 python3 ./tests/test_networkit.py

echo All done!


