"""Regression tests pinning down the tree "Section" node-index convention.

neuroh5 stores each tree's per-section point membership as a flat
'sections' array: [num_sections, (num_nodes_in_section, node_idx...) x
num_sections]. node_idx is 0-based: it indexes directly into the tree's
0-based x/y/z/radius/... point arrays (see py_build_tree_value in
python/neuroh5/iomodule.cc), so valid indices for a tree with N points
are 0 .. N-1 inclusive, with every index appearing in exactly one
section.

This was not always consistent. commit 6f26369 ("bug fix in
py_build_tree_value") changed the reader to require 1-based indices
(1 .. N, with `node_idx - 1` used to index the point arrays), reasoning
from how neurotrees_import's SWC-file path builds section data (SWC
point ids are conventionally 1-based). But every real tree file actually
in use is written 0-based, and reverting py_build_tree_value to enforce
0-based (matching production data) in turn broke tests/unit/test_trees.py,
whose fixtures had been written with 1-based indices to match that
commit. Both sides -- library and fixtures -- have to agree, and it's the
fixtures that were wrong: see _neuroh5_testing.make_tree, which now
generates 0-based section data.

These tests exist so that a future change to either side can't silently
drift out of sync with the other again: they pin the convention (0-based)
explicitly and verify that data using the *other* convention is rejected
rather than silently misread.
"""
import subprocess
import sys
from pathlib import Path

from mpi4py import MPI

from neuroh5.io import append_cell_trees, read_trees

from _neuroh5_testing import create_populations_file, make_tree, trees_equal

COMM = MPI.COMM_WORLD
PROBE = str(Path(__file__).resolve().parent / "_invalid_tree_probe.py")


def _run_probe(tmp_path, **extra_args):
    args = [sys.executable, PROBE, "--path", str(tmp_path / "probe.h5")]
    for key, value in extra_args.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    return subprocess.run(args, capture_output=True, text=True, timeout=60)


def test_zero_based_tree_round_trips(h5_path):
    # The full valid index range for an N-point tree is 0 .. N-1 -- this
    # exercises both ends of it (index 0 in the first section, N-1 -- the
    # highest legal index -- in the last), which is exactly the boundary
    # commit 6f26369 got wrong for the 1-based interpretation (it wrote one
    # slot past the end of an N-sized buffer for index N).
    n_pts = 6
    tree = make_tree(n_pts, gid=0)
    create_populations_file(h5_path, [("GC", 0, 1, 0)])
    append_cell_trees(h5_path, "GC", {0: tree}, comm=COMM)

    g, n_nodes = read_trees(h5_path, "GC", comm=COMM)
    got = dict(g)
    assert n_nodes == 1
    assert trees_equal(got, {0: tree})


def test_probe_agrees_unshifted_tree_is_valid(tmp_path):
    # Sanity-checks the probe script itself against the same convention,
    # so the rejection tests below can be trusted to be testing the shift,
    # not a broken probe.
    result = _run_probe(tmp_path, shift=0)
    assert result.returncode == 0, result.stderr
    assert "OK 0" in result.stdout


def test_one_based_tree_is_rejected(tmp_path):
    # Reproduces the exact mistake flagged as plausible: a tree imported
    # from a standard (1-based) SWC file via neurotrees_import without the
    # -n -1 offset needed to rebase it to 0. Every node index ends up one
    # higher than it should be, so the highest index in the tree becomes N
    # (one past the valid 0 .. N-1 range) -- this must be rejected, not
    # silently accepted with every point assigned to the wrong section.
    result = _run_probe(tmp_path, shift=1)
    assert "OK 0" not in result.stdout, (
        f"1-based (shift=+1) section indices were silently accepted: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.returncode != 0


def test_index_shifted_negative_is_rejected(tmp_path):
    # The opposite direction: indices shifted below 0 (as NODE_IDX_T is
    # unsigned, this wraps to a huge value) must also be rejected rather
    # than accepted or, worse, used to write out of bounds.
    result = _run_probe(tmp_path, shift=-1)
    assert "OK 0" not in result.stdout, (
        f"out-of-range (shift=-1) section indices were silently accepted: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.returncode != 0
