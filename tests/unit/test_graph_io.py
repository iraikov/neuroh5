import numpy as np
from mpi4py import MPI

from neuroh5.io import (
    write_graph,
    append_graph,
    read_graph,
    read_graph_info,
    read_graph_selection,
)

from _neuroh5_testing import create_populations_file, make_edges, edges_equal, normalize_edge_attrs

COMM = MPI.COMM_WORLD


def _make_graph_file(h5_path, dst_count=8, src_count=10, with_attrs=False, seed=0):
    create_populations_file(
        h5_path, [("PopA", 0, src_count, 0), ("PopB", src_count, dst_count, 1)],
        pop_combs=[(0, 1)],
    )
    edges = make_edges(src_count, dst_count, 0, src_count, seed=seed, with_attrs=with_attrs)
    write_graph(h5_path, "PopA", "PopB", edges, comm=COMM)
    return edges


def test_write_read_graph_round_trip(h5_path):
    expected = _make_graph_file(h5_path)

    g, attr_info = read_graph(h5_path, comm=COMM)
    got = dict(g["PopB"]["PopA"])

    assert edges_equal(got, expected)
    assert attr_info == {"PopB": {"PopA": {}}}


def test_write_read_graph_with_attributes(h5_path):
    expected = _make_graph_file(h5_path, with_attrs=True)

    g, attr_info = read_graph(h5_path, namespaces=["Synapses"], comm=COMM)
    got = dict(g["PopB"]["PopA"])
    name_index = attr_info["PopB"]["PopA"]["Synapses"]
    got = normalize_edge_attrs(got, name_index)

    assert edges_equal(got, expected, with_attrs=True)
    assert name_index == {"weight": 0, "syn_id": 1}


def test_read_graph_info_without_namespaces_argument(h5_path):
    # Regression test: read_graph_info(path, comm=comm) with the namespaces
    # argument omitted entirely used to segfault the whole process --
    # py_read_graph_info called PyList_Check on the raw (NULL-when-omitted)
    # PyObject* before the existing, correctly-NULL-guarded loop further
    # down ever ran. Fixed by guarding that check the same way. This must
    # be able to run in-process (not a subprocess) precisely because it's
    # asserting the crash is gone.
    _make_graph_file(h5_path)

    info = read_graph_info(h5_path, comm=COMM)

    assert ("PopA", "PopB") in info


def test_read_graph_info_with_explicit_namespaces(h5_path):
    _make_graph_file(h5_path, with_attrs=True)

    info = read_graph_info(h5_path, namespaces=["Synapses"], comm=COMM)

    assert ("PopA", "PopB") in info


def test_read_graph_selection(h5_path):
    expected = _make_graph_file(h5_path, dst_count=12, src_count=10)
    selection = [10, 15, 21]

    sel, _ = read_graph_selection(h5_path, selection, comm=COMM)
    got = dict(sel["PopB"]["PopA"])

    assert set(got.keys()) == set(selection)
    assert edges_equal(got, {k: expected[k] for k in selection})


def test_append_graph_on_freshly_created_projection(h5_path):
    # append_graph builds its own extendable datasets when the projection
    # doesn't exist yet, so two successive append_graph calls on a fresh
    # file work. (append_graph on a write_graph-created projection does
    # NOT work -- see test_append_graph_after_write_graph_is_broken below.)
    create_populations_file(h5_path, [("PopA", 0, 10, 0), ("PopB", 10, 8, 1)],
                             pop_combs=[(0, 1)])

    first = {"PopB": {"PopA": {10: (np.array([0, 1, 2], dtype=np.uint32), {})}}}
    append_graph(h5_path, first, comm=COMM)

    second = {"PopB": {"PopA": {11: (np.array([3, 4], dtype=np.uint32), {})}}}
    append_graph(h5_path, second, comm=COMM)

    g, _ = read_graph(h5_path, comm=COMM)
    got = dict(g["PopB"]["PopA"])
    assert edges_equal(got, {10: (np.array([0, 1, 2], dtype=np.uint32), {}),
                             11: (np.array([3, 4], dtype=np.uint32), {})})


def test_append_graph_after_write_graph(h5_path):
    # Regression test: append_graph used to abort the whole process with an
    # HDF5 "dimension cannot exceed the existing maximal size" error when
    # appending to a projection originally created by write_graph, because
    # write_projection.cc created the DST_BLK_IDX/DST_BLK_PTR/DST_PTR/
    # SRC_IDX datasets (and, separately, the edge-attribute datasets via
    # write_edge_attribute) as fixed-size/non-extendable, while
    # append_projection.cc always assumes it can H5Dset_extent them. Fixed
    # by giving all of those datasets H5S_UNLIMITED maxdims + chunked
    # layout at write time, matching what append_projection.cc /
    # create_edge_attribute_datasets already do.
    #
    # The destination population has 5 slots (gids 6..10); write_graph only
    # covers 4 of them (6..9) and append_graph adds edges for the 5th (10),
    # a gid write_graph never touched -- as opposed to appending more edges
    # onto an already-written gid, which legitimately accumulates rather
    # than replacing, and would be testing something different.
    src_count, dst_count = 6, 5
    create_populations_file(
        h5_path, [("PopA", 0, src_count, 0), ("PopB", src_count, dst_count, 1)],
        pop_combs=[(0, 1)],
    )
    expected = make_edges(src_count, dst_count - 1, 0, src_count, seed=0, with_attrs=True)
    write_graph(h5_path, "PopA", "PopB", expected, comm=COMM)

    new_gid = src_count + dst_count - 1
    more = {"PopB": {"PopA": {
        new_gid: (np.array([0, 1], dtype=np.uint32),
                  {"Synapses": {"weight": np.array([0.9, 0.8], dtype=np.float32),
                                "syn_id": np.array([501, 502], dtype=np.uint32)}}),
    }}}
    append_graph(h5_path, more, comm=COMM)

    g, attr_info = read_graph(h5_path, namespaces=["Synapses"], comm=COMM)
    got = dict(g["PopB"]["PopA"])
    name_index = attr_info["PopB"]["PopA"]["Synapses"]
    got = normalize_edge_attrs(got, name_index)

    combined = dict(expected)
    combined[new_gid] = more["PopB"]["PopA"][new_gid]
    assert set(got.keys()) == set(combined.keys())
    assert edges_equal(got, combined, with_attrs=True)
