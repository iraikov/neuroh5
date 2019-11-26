#!/bin/bash

#SBATCH -J neuroh5_scatter_read_trees_test           # Job name
#SBATCH -o ./scatter_read_trees_test.o%j       # Name of stdout output file
#SBATCH -e ./scatter_read_trees_test.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 128             # Total # of nodes 
#SBATCH -n 7168            # Total # of mpi tasks
#SBATCH -t 00:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module unload python2 
module swap intel intel/18.0.5
module load python3
module load phdf5/1.8.16

set -x

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

cd $SLURM_SUBMIT_DIR

export I_MPI_EXTRA_FILESYSTEM=enable
export I_MPI_EXTRA_FILESYSTEM_LIST=lustre
export I_MPI_ADJUST_ALLGATHER=4
export I_MPI_ADJUST_ALLGATHERV=4
export I_MPI_ADJUST_ALLTOALL=4

export PYTHON=`which python3`

ibrun env PYTHONPATH=$PYTHONPATH $PYTHON ./tests/test_scatter_read_trees.py



