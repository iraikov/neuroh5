#!/bin/bash

#SBATCH -J neuroh5_scatter_attrs_test           # Job name
#SBATCH -o ./scatter_attrs_test.o%j       # Name of stdout output file
#SBATCH -e ./scatter_attrs_test.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 32             # Total # of nodes 
#SBATCH -n 1536            # Total # of mpi tasks
#SBATCH -t 00:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5

set -x

data_prefix=$SCRATCH/dentate/Full_Scale_Control
export data_prefix

cd $SLURM_SUBMIT_DIR

#export I_MPI_ADJUST_ALLTOALLV=1
#export I_MPI_DYNAMIC_CONNECTION=0

input_path=${data_prefix}/DG_Cells_Full_Scale_20190512.h5
input_path=${data_prefix}/DGC_forest_reindex_20181222_compressed.h5

ibrun ./build/neurotrees_scatter_read -a -n "Synapse Attributes" -i 512  ${input_path}
      



