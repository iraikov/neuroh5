import pytest
from mpi4py import MPI

from neuroh5.io import write_cell_attributes

from _neuroh5_testing import create_populations_file, make_cell_attrs


def _make_attr_file(h5_path, pop_count, seed=0):
    create_populations_file(h5_path, [("GC", 0, pop_count, 0)])
    attrs = make_cell_attrs(0, pop_count, seed=seed)
    write_cell_attributes(h5_path, "GC", attrs, namespace="Test", comm=MPI.COMM_WORLD)


# 20 cells never divides evenly across 3 ranks (7/7/6) -- the main edge
# case for the round-robin scatter distribution.
POP_COUNT, SEED = 20, 11


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_cell_attributes(h5_path, mpi_worker, nranks):
    _make_attr_file(h5_path, POP_COUNT, seed=SEED)

    result = mpi_worker("cell_attr_worker", nranks, [
        "--scenario", "scatter", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 3])
def test_bcast_cell_attributes(h5_path, mpi_worker, nranks):
    _make_attr_file(h5_path, POP_COUNT, seed=SEED)

    result = mpi_worker("cell_attr_worker", nranks, [
        "--scenario", "bcast", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_cell_attribute_selection(h5_path, mpi_worker, nranks):
    _make_attr_file(h5_path, POP_COUNT, seed=SEED)

    result = mpi_worker("cell_attr_worker", nranks, [
        "--scenario", "scatter_selection", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_append_cell_attributes_multi_rank(h5_path, mpi_worker, nranks):
    # Regression test: create_cell_attribute_datasets (and create_cell_index,
    # used internally by the tree append path with the same pattern) used to
    # have every I/O rank independently exists_dataset-check-then-create the
    # attribute namespace's groups/datasets on a collectively-open parallel
    # file; racing/disagreeing ranks issued a mismatched sequence of
    # collective H5Gcreate/H5Dcreate2 calls and corrupted the file. Targets
    # a freshly created file (no "Test" namespace group yet) with every rank
    # acting as an I/O rank (io_size defaults to nranks) -- the configuration
    # that used to race.
    create_populations_file(h5_path, [("GC", 0, POP_COUNT, 0)])

    result = mpi_worker("cell_attr_worker", nranks, [
        "--scenario", "append", "--path", h5_path,
        "--pop-start", "0", "--pop-count", str(POP_COUNT), "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]
