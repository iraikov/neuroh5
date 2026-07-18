import numpy as np
from mpi4py import MPI

from neuroh5.io import (
    write_cell_attributes,
    append_cell_attributes,
    read_cell_attributes,
    read_cell_attribute_info,
    read_cell_attribute_selection,
)

from _neuroh5_testing import create_populations_file, make_cell_attrs, cell_attrs_equal

COMM = MPI.COMM_WORLD


def _make_attr_file(h5_path, pop_count=20, seed=0):
    create_populations_file(h5_path, [("GC", 0, pop_count, 0)])
    attrs = make_cell_attrs(0, pop_count, seed=seed)
    write_cell_attributes(h5_path, "GC", attrs, namespace="Test", comm=COMM)
    return attrs


def test_write_read_cell_attributes_round_trip(h5_path):
    expected = _make_attr_file(h5_path)

    got = dict(read_cell_attributes(h5_path, "GC", namespace="Test", comm=COMM))

    assert cell_attrs_equal(got, expected)


def test_read_cell_attribute_info(h5_path):
    _make_attr_file(h5_path)

    info = read_cell_attribute_info(h5_path, ["GC"], comm=COMM)

    assert set(info["GC"]["Test"]) == {"a", "b"}


def test_read_cell_attribute_selection(h5_path):
    expected = _make_attr_file(h5_path, pop_count=20)
    selection = [0, 7, 19]

    got = dict(read_cell_attribute_selection(h5_path, "GC", selection,
                                             namespace="Test", comm=COMM))

    assert set(got.keys()) == set(selection)
    assert cell_attrs_equal(got, {k: expected[k] for k in selection})


def test_append_cell_attributes_adds_new_gids(h5_path):
    # Population range must cover both the initially-written gids and the
    # ones added later via append -- allocate it up front (12 slots), but
    # only write attributes for the first 10 initially.
    create_populations_file(h5_path, [("GC", 0, 12, 0)])
    expected = make_cell_attrs(0, 10)
    write_cell_attributes(h5_path, "GC", expected, namespace="Test", comm=COMM)

    more = {
        10: {"a": np.array([1, 2], dtype=np.uint32), "b": np.array([0.5], dtype=np.float32)},
        11: {"a": np.array([3], dtype=np.uint32), "b": np.array([0.25], dtype=np.float32)},
    }
    append_cell_attributes(h5_path, "GC", more, namespace="Test", comm=COMM)

    got = dict(read_cell_attributes(h5_path, "GC", namespace="Test", comm=COMM))
    expected.update({k: v for k, v in more.items()})
    assert cell_attrs_equal(got, expected)
