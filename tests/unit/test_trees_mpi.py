import pytest
from mpi4py import MPI

from neuroh5.io import append_cell_trees

from _neuroh5_testing import create_populations_file, make_trees


def _make_tree_file(h5_path, pop_count, seed=0):
    create_populations_file(h5_path, [("GC", 0, pop_count, 0)])
    trees = make_trees(0, pop_count, seed=seed)
    append_cell_trees(h5_path, "GC", trees, comm=MPI.COMM_WORLD)


# 11 trees never divides evenly across 3 or 4 ranks -- the main edge case
# for the round-robin scatter distribution.
POP_COUNT, SEED = 11, 3


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_trees(h5_path, mpi_worker, nranks):
    _make_tree_file(h5_path, POP_COUNT, seed=SEED)

    result = mpi_worker("tree_worker", nranks, [
        "--scenario", "scatter", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_tree_selection(h5_path, mpi_worker, nranks):
    _make_tree_file(h5_path, POP_COUNT, seed=SEED)

    result = mpi_worker("tree_worker", nranks, [
        "--scenario", "scatter_selection", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_append_cell_trees_multi_rank(h5_path, mpi_worker, nranks):
    # Regression test: create_cell_index (used internally by append_trees)
    # used to have every I/O rank independently exists_dataset-check the
    # tree population's cell-index group/dataset on a collectively-open
    # parallel file with no rank guard at all; racing/disagreeing ranks
    # issued a mismatched sequence of collective H5Gcreate/H5Dcreate2 calls
    # and corrupted the file. Targets a freshly created file (no "GC"
    # Trees group yet) with every rank acting as an I/O rank (io_size
    # defaults to nranks) -- the configuration that used to race.
    create_populations_file(h5_path, [("GC", 0, POP_COUNT, 0)])

    result = mpi_worker("tree_worker", nranks, [
        "--scenario", "append", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]
