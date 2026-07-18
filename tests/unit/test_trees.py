from mpi4py import MPI

from neuroh5.io import append_cell_trees, read_trees, read_tree_selection

from _neuroh5_testing import create_populations_file, make_trees, trees_equal

COMM = MPI.COMM_WORLD


def _make_tree_file(h5_path, pop_count=8, seed=0):
    create_populations_file(h5_path, [("GC", 0, pop_count, 0)])
    trees = make_trees(0, pop_count, seed=seed)
    append_cell_trees(h5_path, "GC", trees, comm=COMM)
    return trees


def test_append_read_trees_round_trip(h5_path):
    expected = _make_tree_file(h5_path)

    g, n_nodes = read_trees(h5_path, "GC", comm=COMM)
    got = dict(g)

    assert n_nodes == len(expected)
    assert trees_equal(got, expected)


def test_read_tree_selection(h5_path):
    expected = _make_tree_file(h5_path, pop_count=10)
    selection = [0, 4, 9]

    sel, _ = read_tree_selection(h5_path, "GC", selection, comm=COMM)
    got = dict(sel)

    assert set(got.keys()) == set(selection)
    assert trees_equal(got, {k: expected[k] for k in selection})


def test_read_tree_selection_namespaces_keyword(h5_path):
    # Regression test: py_read_tree_selection's kwlist named its 5th/6th
    # keyword args "mask" then "namespaces", but bound them to the wrong C
    # variables (&py_attr_name_spaces then &py_mask, i.e. swapped) -- so
    # calling with namespaces=[...] actually populated the mask check
    # (a set-of-strings) and vice versa. A caller passing namespaces=[]
    # here used to raise "attribute name spaces argument is not a list"
    # (or, for a non-empty list, "argument mask must be a set of strings").
    expected = _make_tree_file(h5_path, pop_count=5)
    selection = [0, 2]

    sel, _ = read_tree_selection(h5_path, "GC", selection, namespaces=[], comm=COMM)
    got = dict(sel)
    assert trees_equal(got, {k: expected[k] for k in selection})


def test_read_tree_selection_mask_keyword(h5_path):
    expected = _make_tree_file(h5_path, pop_count=5)
    selection = [0, 2]

    # A mask naming attributes that don't exist on trees is accepted (it's
    # just ignored) -- the point here is only that passing mask= doesn't
    # raise "argument mask must be a set of strings" the way it did when
    # this keyword was cross-wired to the namespaces list.
    sel, _ = read_tree_selection(h5_path, "GC", selection, mask={"x"}, comm=COMM)
    got = dict(sel)
    assert set(got.keys()) == set(selection)
