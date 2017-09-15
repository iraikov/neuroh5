#include "debug.hh"

#include "neuroh5_types.hh"
#include "cell_populations.hh"

#include "read_projection.hh"
#include "read_graph.hh"
#include "scatter_read_graph.hh"
#include "cell_populations.hh"
#include "projection_names.hh"
#include "validate_edge_list.hh"

#include "hdf5.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include <mpi.h>

using namespace std;
using namespace neuroh5;

/*****************************************************************************
 * Append src/dst node pairs to a list of edges
 *****************************************************************************/

int filter_edge_list
(
 prj_tuple_t&              projection,
 const set<NODE_IDX_T>&    src_selection,
 const set<NODE_IDX_T>&    dst_selection,
 vector<NODE_IDX_T>&       edge_list
 )
{
  int ierr = 0;
  const vector <NODE_IDX_T>& src_vector = get<0>(projection);
  const vector <NODE_IDX_T>& dst_vector = get<1>(projection);
  
  for (size_t i = 0; i<src_vector.size(); i++)
    {
      NODE_IDX_T src = src_vector[i];
      NODE_IDX_T dst = dst_vector[i];
      if (src_selection.empty() || (src_selection.find(src) != src_selection.end()))
        {
          edge_list.push_back(src);
          edge_list.push_back(dst);
        }
    }

  return ierr;
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  string input_file_name, src_pop_name, dst_pop_name;
  vector <string> edge_attr_name_spaces;
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  // parse arguments

  if (argc < 3) 
    {
      std::cout << "Usage: reader <FILE> <SRC> <DST> <SELECTION> ..." << std::endl;
      exit(1);
    }

  input_file_name = string(argv[1]);
  src_pop_name = string(argv[2]);
  dst_pop_name = string(argv[3]);

  // determine src and dst node selections
  set <NODE_IDX_T> src_selection;
  set <NODE_IDX_T> dst_selection;
  ifstream srcfile(argv[3]);
  string line;
  size_t lnum = 0;
  // reads node to rank assignment from file
  while (getline(srcfile, line))
    {
      istringstream iss(line);
      NODE_IDX_T n;
      
      assert (iss >> n);
      src_selection.insert(n);
      lnum++;
    }
  
  srcfile.close();

  ifstream dstfile(argv[4]);

  // reads node to rank assignment from file
  while (getline(dstfile, line))
    {
      istringstream iss(line);
      NODE_IDX_T n;
      
      assert (iss >> n);
      dst_selection.insert(n);
      lnum++;
    }
  dstfile.close();

  assert (!((src_selection.size() == 0) && (dst_selection.size() == 0)));
  // read the population info
  set< pair<pop_t, pop_t> > pop_pairs;
  assert(cell::read_population_combos(MPI_COMM_WORLD, input_file_name, pop_pairs) >= 0);

  size_t total_num_nodes;
  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
  assert(cell::read_population_ranges(MPI_COMM_WORLD, input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

  vector< pair<pop_t, string> > pop_labels;
  assert(cell::read_population_labels(MPI_COMM_WORLD, input_file_name, pop_labels) >= 0);
  
  vector< pair<string, string> > prj_names;
  assert(graph::read_projection_names(MPI_COMM_WORLD, input_file_name, prj_names) >= 0);
      
  vector<NODE_IDX_T> edge_list;
  vector<prj_tuple_t> prj_vector;
  vector < map <string, vector < vector<string> > > >  edge_attr_names_vector;

  // read the edges
  for (size_t i = 0; i < prj_names.size(); i++)
    {
      size_t total_prj_num_edges = 0, local_prj_num_edges = 0;

      if ((src_pop_name.compare(prj_names[i].first) == 0) &&
          (dst_pop_name.compare(prj_names[i].second) == 0))
        {
          printf("Reading projection %lu (%s -> %s)\n", i, prj_names[i].first.c_str(), prj_names[i].second.c_str());

          uint32_t dst_pop_idx = 0, src_pop_idx = 0;
          bool src_pop_set = false, dst_pop_set = false;
      
          for (size_t i=0; i< pop_labels.size(); i++)
            {
              if (src_pop_name == get<1>(pop_labels[i]))
                {
                  src_pop_idx = get<0>(pop_labels[i]);
                  src_pop_set = true;
                }
              if (dst_pop_name == get<1>(pop_labels[i]))
                {
                  dst_pop_idx = get<0>(pop_labels[i]);
                  dst_pop_set = true;
                }
            }
          assert(dst_pop_set && src_pop_set);

          NODE_IDX_T dst_start = pop_vector[dst_pop_idx].start;
          NODE_IDX_T src_start = pop_vector[src_pop_idx].start;

          assert(graph::read_projection(MPI_COMM_WORLD, input_file_name,
                                        pop_ranges, pop_pairs,
                                        prj_names[i].first, prj_names[i].second,
                                        dst_start, src_start, 
                                        edge_attr_name_spaces, 
                                        prj_vector, edge_attr_names_vector,
                                        local_prj_num_edges, total_prj_num_edges) >= 0);

          
          // filter/append to the edge list
          assert(filter_edge_list(prj_vector[0], src_selection, dst_selection, edge_list) >= 0);
        }
    }
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (edge_list.size() > 0) 
    {
      ofstream outfile;
      stringstream outfilename;
      assert(edge_list.size()%2 == 0);

      outfilename << string(input_file_name) << "." << rank << ".subgraph";
      outfile.open(outfilename.str());

      for (size_t i = 0, k = 0; i < edge_list.size()-1; i+=2, k++)
        {
          outfile << k << " " << edge_list[i] << " " << edge_list[i+1] << std::endl;
        }
      outfile.close();

    }

  MPI_Finalize();
  return 0;
}
