#!/bin/bash

## Run the reader to generated *.edges files per projection and per rank
mpirun -n 4 ./reader/src/reader -a ./data/dentate_test.h5

## Concatenate all results into one file per projection and remove first column 
cat data/dentate_test.h5.0.[0-9]*.edges | cut -d' ' -f2,3 | sort -s -k 2n,1n > data/dentate_test.h5.0.test.edges
cat data/dentate_test.h5.1.[0-9]*.edges | cut -d' ' -f2,3 | sort -s -k 2n,1n > data/dentate_test.h5.1.test.edges

## Remove first column from validation data file
cut -f1,2 -d' ' data/dentate_test.h5.0.edges > data/dentate_test.h5.0.base.edges
cut -f1,2 -d' ' data/dentate_test.h5.1.edges > data/dentate_test.h5.1.base.edges

diff -u data/dentate_test.h5.0.test.edges data/dentate_test.h5.0.base.edges
diff -u data/dentate_test.h5.1.test.edges data/dentate_test.h5.1.base.edges
