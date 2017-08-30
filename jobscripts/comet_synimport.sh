#!/bin/bash
#
#SBATCH -J neurograph_synimport
#SBATCH -o ./results/neurograph_synimport.%j.o
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x

prefix=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/
export prefix

forest_connectivity_path=$prefix/DGC_forest_connectivity_20170508.h5
connectivity_output_path=$prefix/dentate_Full_Scale_GC_20170728.h5

for post in GC; do

for pre in MPP LPP MC BC HC AAC HCC NGFC MOPP; do

aprun -n 1008 ./build/neurograph_import -f hdf5:syn -s 128 \
      -d $forest_connectivity_path:/Populations/$post/Connectivity \
      $pre $post $connectivity_output_path

done
done


