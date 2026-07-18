"""Regression tests for the C++-exception-to-Python-exception boundary.

neuroh5.io's py_* entry points are hand-written CPython C API functions
wrapping C++ code that uses throw_assert/throw_assert_nomsg (see
include/throw_assert.hh), which throw a plain AssertionFailureException
(a std::exception subclass) on failure. Before python/neuroh5/iomodule.cc
wrapped every module_methods[] entry with a NEUROH5_PY_GUARD-generated
try/catch, any such failure unwound uncaught into CPython's C frames --
undefined behavior that reliably crashed the whole process
(std::terminate()/SIGABRT, or sometimes a straight segfault) instead of
raising a normal, catchable Python exception. These tests must run
in-process (not via a subprocess) precisely because they are asserting
that the crash is gone.
"""
from mpi4py import MPI

from neuroh5.io import read_population_ranges, read_cell_attribute_info, read_graph_info

COMM = MPI.COMM_WORLD


def test_nonexistent_file_raises_runtime_error_not_crash(tmp_path):
    missing = str(tmp_path / "does_not_exist.h5")

    try:
        read_population_ranges(missing, comm=COMM)
    except RuntimeError as e:
        assert "Assertion" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")


def test_read_cell_attribute_info_nonexistent_file(tmp_path):
    missing = str(tmp_path / "does_not_exist.h5")

    try:
        read_cell_attribute_info(missing, ["GC"], comm=COMM)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected a RuntimeError")


def test_read_graph_info_nonexistent_file(tmp_path):
    missing = str(tmp_path / "does_not_exist.h5")

    try:
        read_graph_info(missing, namespaces=[], comm=COMM)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected a RuntimeError")


def test_process_survives_multiple_caught_failures(tmp_path):
    # A single stray uncaught exception could plausibly "work" once by
    # accident depending on stack layout; call several different entry
    # points back-to-back to make sure the guard is doing its job
    # consistently rather than the process merely surviving by luck.
    missing = str(tmp_path / "does_not_exist.h5")
    caught = 0
    for _ in range(5):
        try:
            read_population_ranges(missing, comm=COMM)
        except RuntimeError:
            caught += 1
    assert caught == 5
