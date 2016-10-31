"""Python routines to import connectivity data in the neurograph storage format and bindings to the neurograph reader functions.


The neurograph library implements an HDF5-based format for storing
connectivity information of large neural networks, perform parallel
graph partitioning and analysis.

neurograph assumes that synaptic connectivity between neurons in
neuronal network models is represented as directed graphs stored as
adjacency lists, where the vertices represent the neurons in the
network and are identified by unsigned integers called unique global
identifiers (gid). 

The importdbs script can be used to import projection data from text
files that contain source, destination pairs and corresponding edge
labels.

Example::
  importdbs import-globals [OPTIONS] POPULATION_FILE CONNECTIVITY_FILE OUTPUTFILE

Creates an initial definition of populations and
connectivity. POPULATION_FILE is a text file containing start index,
size, index for each population. CONNECTIVITY_FILE contains the valid
population combinations in the form projection label, source
population index, destination population index. OUTPUTFILE is the HDF5
connectivity file.

Example::
  importdbs import-lsn [OPTIONS] SOURCE DEST GROUPNAME [INPUTFILES]... OUTPUTFILE

Imports connectivity from text files in the LAYER-SEGMENT-INDEX format. This format consists of the following fields:

  SRC DEST WEIGHT LAYER SECTION SECTION-POINT

where SRC: is the source neuron index DEST: is the destination neuron index WEIGHT: is the weight of that edge SECTION: section index on the destination neuron SECTION-POINT: point index on the destination neuron

Argument SOURCE is the name of the source population. Argument DEST is
the name of the destination population. GROUPNAME is the HDF5 dataset
name for that projection. INPUTFILES are the input text files, and
OUTPUTFILE is the HDF5 file with connectivity.

Example::
  importdbs import-ltdist [OPTIONS] SOURCE DEST GROUPNAME [INPUTFILES]... OUTPUTFILE

Imports connectivity from text files in the LONGITUDINAL-TRANSVERSE
format. This format consists of the following fields:

  SRC DEST LONGITUDINAL-DISTANCE TRANSVERSE-DISTANCE

where SRC: is the source neuron index DEST: is the destination neuron
index LONGITUDINAL-DISTANCE: is the longitudinal distance in
micrometers between source and destination neurons
TRANSVERSE-DISTANCE: is the transverse distance in micrometers between
source and destination neurons

"""
