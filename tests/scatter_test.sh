#!/bin/bash

## Run the reader to generated *.edges files per projection and per rank
mpirun -n 4 ./reader/src/scatter -a -i 2 -o ./data/dentate_test ./data/dentate_test.h5

## Concatenate all results into one file per projection and remove first column 
cat data/dentate_test.0.[0-9]*.edges | cut -d' ' -f1,2 | sort -s -k 2n > data/dentate_test.0.test.edges
cat data/dentate_test.1.[0-9]*.edges | cut -d' ' -f1,2 | sort -s -k 2n > data/dentate_test.1.test.edges

## Remove first column from validation data file
cut -d' ' -f1,2 data/dentate_test.h5.0.edges > data/dentate_test.0.base.edges
cut -d' ' -f1,2 data/dentate_test.h5.1.edges > data/dentate_test.1.base.edges

diff -u data/dentate_test.0.test.edges data/dentate_test.0.base.edges
diff -u data/dentate_test.1.test.edges data/dentate_test.1.base.edges
