from mpi4py import MPI

from neuroh5.io import read_population_ranges, read_population_names, read_projection_names

from _neuroh5_testing import create_populations_file


def test_read_population_ranges(h5_path):
    create_populations_file(h5_path, [("PopA", 0, 10, 0), ("PopB", 10, 8, 1)])

    ranges, total = read_population_ranges(h5_path, comm=MPI.COMM_WORLD)

    assert ranges == {"PopA": (0, 10), "PopB": (10, 8)}
    assert total == 18


def test_read_population_names_empty_before_any_attributes_written(h5_path):
    create_populations_file(h5_path, [("PopA", 0, 10, 0)])

    # /Populations/<name> groups only appear once attributes/trees are
    # written for that population -- a freshly created file has none.
    assert read_population_names(h5_path, comm=MPI.COMM_WORLD) == []


def test_read_population_names_after_write(h5_path):
    import numpy as np
    from neuroh5.io import write_cell_attributes

    create_populations_file(h5_path, [("PopA", 0, 5, 0)])
    write_cell_attributes(
        h5_path, "PopA", {0: {"a": np.array([1], dtype=np.uint32)}},
        namespace="Test", comm=MPI.COMM_WORLD,
    )

    assert read_population_names(h5_path, comm=MPI.COMM_WORLD) == ["PopA"]


def test_read_projection_names(h5_path):
    import numpy as np
    from neuroh5.io import write_graph

    create_populations_file(h5_path, [("PopA", 0, 5, 0), ("PopB", 5, 5, 1)],
                             pop_combs=[(0, 1)])
    write_graph(h5_path, "PopA", "PopB",
                {5: (np.array([0], dtype=np.uint32), {})}, comm=MPI.COMM_WORLD)

    assert read_projection_names(h5_path, comm=MPI.COMM_WORLD) == [("PopA", "PopB")]
