#!/bin/bash

#SBATCH -J neuroh5_scatter_read_trees_test           # Job name
#SBATCH -o ./scatter_read_trees_test.o%j       # Name of stdout output file
#SBATCH -e ./scatter_read_trees_test.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 40             # Total # of nodes 
#SBATCH -n 2240           # Total # of mpi tasks
#SBATCH -t 00:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5

set -x

export NEURONROOT=$SCRATCH/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

cd $SLURM_SUBMIT_DIR

export I_MPI_ADJUST_ALLGATHER=4
export I_MPI_ADJUST_ALLGATHERV=4
export I_MPI_ADJUST_ALLTOALL=4


ibrun -n 24 env python3 ./tests/test_scatter_read_trees.py



