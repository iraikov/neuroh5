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

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

set -x

cd $SLURM_SUBMIT_DIR

data_prefix=$SCRATCH/striped/dentate/Slice
export data_prefix

input_path=${data_prefix}/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210203_compressed.h5

ibrun -n 24 python3 ./tests/test_scatter_read_syn_attrs.py --syn-path=$input_path --io-size=8
      



