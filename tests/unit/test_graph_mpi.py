import pytest
from mpi4py import MPI

from neuroh5.io import write_graph

from _neuroh5_testing import create_populations_file, make_edges


def _make_graph_file(h5_path, dst_count, src_count, with_attrs=False, seed=0):
    create_populations_file(
        h5_path, [("PopA", 0, src_count, 0), ("PopB", src_count, dst_count, 1)],
        pop_combs=[(0, 1)],
    )
    edges = make_edges(src_count, dst_count, 0, src_count, seed=seed, with_attrs=with_attrs)
    write_graph(h5_path, "PopA", "PopB", edges, comm=MPI.COMM_WORLD)


# 20 destinations never divides evenly across 3 ranks (7/7/6) -- this is the
# main edge case: uneven round-robin distribution of the DBS destination set.
DST_COUNT, SRC_COUNT, SEED = 20, 15, 7


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_graph(h5_path, mpi_worker, nranks):
    _make_graph_file(h5_path, DST_COUNT, SRC_COUNT, seed=SEED)

    result = mpi_worker("graph_worker", nranks, [
        "--scenario", "scatter", "--path", h5_path,
        "--dst-start", str(SRC_COUNT), "--dst-count", str(DST_COUNT),
        "--src-start", "0", "--src-count", str(SRC_COUNT),
        "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_graph_with_attributes(h5_path, mpi_worker, nranks):
    _make_graph_file(h5_path, DST_COUNT, SRC_COUNT, with_attrs=True, seed=SEED)

    result = mpi_worker("graph_worker", nranks, [
        "--scenario", "scatter", "--path", h5_path,
        "--dst-start", str(SRC_COUNT), "--dst-count", str(DST_COUNT),
        "--src-start", "0", "--src-count", str(SRC_COUNT),
        "--seed", str(SEED), "--with-attrs",
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 3])
def test_bcast_graph(h5_path, mpi_worker, nranks):
    _make_graph_file(h5_path, DST_COUNT, SRC_COUNT, seed=SEED)

    result = mpi_worker("graph_worker", nranks, [
        "--scenario", "bcast", "--path", h5_path,
        "--dst-start", str(SRC_COUNT), "--dst-count", str(DST_COUNT),
        "--src-start", "0", "--src-count", str(SRC_COUNT),
        "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_scatter_read_graph_selection(h5_path, mpi_worker, nranks):
    # Regression test: scatter_read_graph_selection used to abort at any
    # comm size > 1 ("mismatch in projection edge count") because
    # all_selections (every rank's selection, Allgatherv'd together) was
    # never de-duplicated before driving the disk read, so a gid asked for
    # by N ranks got read and appended N times. Fixed in
    # scatter_read_graph_selection.cc (dedupe all_selections) and
    # append_rank_edge_map_selection.cc (edge-count accounting was also
    # inconsistent whenever a destination legitimately maps to more than
    # one rank).
    _make_graph_file(h5_path, DST_COUNT, SRC_COUNT, seed=SEED)

    result = mpi_worker("graph_worker", nranks, [
        "--scenario", "scatter_selection", "--path", h5_path,
        "--dst-start", str(SRC_COUNT), "--dst-count", str(DST_COUNT),
        "--src-start", "0", "--src-count", str(SRC_COUNT),
        "--seed", str(SEED),
    ])

    assert result["ok"], result["errors"]


@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_append_graph_multi_rank(h5_path, mpi_worker, nranks):
    # Regression test: append_graph used to deadlock (hang the whole MPI
    # job) or silently drop non-root ranks' contributions whenever more
    # than one rank acted as an I/O rank (io_size > 1). Reproduced even on
    # a fresh file with two successive append_graph calls -- unrelated to
    # write_graph. Fixed in append_projection.cc: dst_blk_idx/dst_blk_ptr/
    # dst_ptr (small block/pointer metadata) are now gathered to the root
    # I/O rank via MPI_Gatherv and written by it alone, instead of every
    # I/O rank independently extending and collectively writing its own
    # hyperslab -- multiple ranks collectively writing disjoint hyperslabs
    # of what's typically a single small HDF5 chunk reliably deadlocked.
    # src_idx (large) keeps its original per-rank distributed collective
    # write. The worker forces io_size == nranks so every rank is an I/O
    # rank -- the configuration that used to hang.
    create_populations_file(
        h5_path, [("PopA", 0, SRC_COUNT, 0), ("PopB", SRC_COUNT, DST_COUNT, 1)],
        pop_combs=[(0, 1)],
    )

    result = mpi_worker("graph_worker", nranks, [
        "--scenario", "append", "--path", h5_path,
        "--dst-start", str(SRC_COUNT), "--dst-count", str(DST_COUNT),
        "--src-start", "0", "--src-count", str(SRC_COUNT),
        "--seed", str(SEED), "--io-size", str(nranks),
    ])

    assert result["ok"], result["errors"]
