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
