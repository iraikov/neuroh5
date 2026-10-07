import json

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from neuroh5.io import write_graph

from _neuroh5_testing import create_populations_file


# Two projections onto PopD with different sets of destination gids:
# PopA -> PopD covers about two thirds of PopD, PopB -> PopD only a fifth,
# so the two generators yield very different numbers of items and refill
# their caches at different steps. The gaps in the destination sets split
# each projection into many blocks, so that every combination of rank
# count and cache size below reads it in more than one window. 61
# destinations never divide evenly across 2, 3 or 4 ranks.
A_START, A_COUNT = 0, 15
B_START, B_COUNT = 15, 12
D_START, D_COUNT = 27, 61
SEED = 11


def _make_projection_file(h5_path, spec_path):
    create_populations_file(
        h5_path,
        [("PopA", A_START, A_COUNT, 0), ("PopB", B_START, B_COUNT, 1),
         ("PopD", D_START, D_COUNT, 2)],
        pop_combs=[(0, 2), (1, 2)],
    )
    rng = np.random.default_rng(SEED)
    spec = {}
    for src_pop, src_start, src_count, keep in (
            ("PopA", A_START, A_COUNT, 0.65), ("PopB", B_START, B_COUNT, 0.2)):
        edges = {}
        for dst in range(D_START, D_START + D_COUNT):
            if rng.random() >= keep:
                continue
            deg = int(rng.integers(1, 6))
            src = rng.integers(src_start, src_start + src_count, deg).astype(np.uint32)
            # Distinct per-edge values, so that attributes read from the
            # wrong edge offset cannot match.
            syn_id = rng.integers(0, 1 << 30, deg).astype(np.uint32)
            edges[dst] = (src, {"Synapses": {"syn_id": syn_id}})
        write_graph(h5_path, src_pop, "PopD", edges, comm=MPI.COMM_WORLD)
        spec[src_pop] = {str(d): [v[0].tolist(), v[1]["Synapses"]["syn_id"].tolist()]
                         for d, v in edges.items()}
    with open(spec_path, "w") as f:
        json.dump(spec, f)


def _num_blocks(h5_path, src_pop):
    with h5py.File(h5_path, "r") as h5:
        return len(h5[f"Projections/PopD/{src_pop}/Edges/Destination Block Pointer"]) - 1


@pytest.mark.parametrize("cache_size", [1, 2])
@pytest.mark.parametrize("nranks", [1, 2, 3, 4])
def test_projection_gen_rank_map_matches_scatter_read_graph(h5_path, tmp_path, mpi_worker,
                                                            nranks, cache_size):
    # Regression test for two defects of NeuroH5ProjectionGen:
    #
    # * Destination gids were assigned to ranks by their position among the
    #   gids with edges in the blocks an I/O rank read, rather than by the
    #   gid itself, so generators for different projections onto the same
    #   population delivered the same gid to different ranks whenever their
    #   sets of destinations with edges differed. Destinations are now
    #   assigned to rank gid % comm_size, as in scatter_read_graph.
    #
    # * read_projection_datasets ignored the requested block window, so the
    #   generator read the whole projection in its first refill regardless
    #   of cache_size. The worker checks the exact number of items yielded
    #   per rank, which depends on the window size.
    spec_path = str(tmp_path / "spec.json")
    _make_projection_file(h5_path, spec_path)
    for src_pop in ("PopA", "PopB"):
        assert _num_blocks(h5_path, src_pop) > cache_size * nranks, (
            f"fixture projection {src_pop} -> PopD fits in a single window")

    result = mpi_worker("projection_gen_worker", nranks, [
        "--path", h5_path, "--spec", spec_path, "--dst-start", str(D_START),
        "--cache-size", str(cache_size),
    ])

    assert result["ok"], result["errors"]
    assert result["nranks"] == nranks
