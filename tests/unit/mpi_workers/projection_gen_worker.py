#!/usr/bin/env python
"""MPI worker comparing the destination-gid-to-rank assignment of
NeuroH5ProjectionGen with that of scatter_read_graph.

The file holds two projections, PopA -> PopD and PopB -> PopD, whose sets
of destination gids with edges differ. For each projection the worker
checks that:

* every destination gid is delivered by NeuroH5ProjectionGen to the same
  rank as scatter_read_graph delivers it, so a gid present in both
  projections receives all of its edges on one rank;
* every rank yields the same number of items from a generator;
* the edges yielded, and their Synapses attributes, equal those written;
* the generator reads the projection in windows of cache_size * comm_size
  blocks: the number of items each rank yields must equal the sum, over
  the windows, of the largest number of gids that any one rank receives
  from that window, plus the final (None, None).

The generators are iterated in lockstep, with one collective call per
step, in the way synaptic weight generation scripts consume them.

Usage: mpirun -n N python projection_gen_worker.py
           --path FILE --spec FILE --dst-start I [--cache-size I]
           --out FILE
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker_common import WorkerResult, MPI  # noqa: E402

from neuroh5.io import NeuroH5ProjectionGen, scatter_read_graph  # noqa: E402


def edge_pairs(src, syn_id):
    """Sorted (source gid, syn_id) pairs of one destination's edges."""
    return sorted(zip(np.asarray(src).tolist(), np.asarray(syn_id).tolist()))


def windowed_item_count(path, src_pop, dst_start, nranks, window):
    """Number of items every rank should yield from a generator that reads
    the projection src_pop -> PopD in windows of the given number of blocks
    and sends each destination gid to rank gid % nranks.

    The destination pointer array ends with a sentinel that the last block
    also counts, so only pointer positions before it denote destinations.
    """
    with h5py.File(path, "r") as h5:
        g = h5[f"Projections/PopD/{src_pop}/Edges"]
        blk_ptr = g["Destination Block Pointer"][:]
        blk_idx = g["Destination Block Index"][:]
        dst_ptr = g["Destination Pointer"][:]
    num_blocks = len(blk_ptr) - 1
    count = 0
    for w in range(0, num_blocks, window):
        per_rank = [0] * nranks
        for b in range(w, min(w + window, num_blocks)):
            for i in range(blk_ptr[b], blk_ptr[b + 1]):
                if i + 1 < len(dst_ptr) and dst_ptr[i + 1] > dst_ptr[i]:
                    gid = dst_start + int(blk_idx[b]) + (i - int(blk_ptr[b]))
                    per_rank[gid % nranks] += 1
        count += max(per_rank)
    return count + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--spec", required=True,
                   help="JSON file: {src_pop: {dst_gid: [[src_gid, ...], [syn_id, ...]]}}")
    p.add_argument("--dst-start", type=int, required=True)
    p.add_argument("--cache-size", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    result = WorkerResult(comm)

    spec = json.loads(Path(args.spec).read_text())
    sources = sorted(spec.keys())
    expected = {s: {int(d): edge_pairs(v[0], v[1]) for d, v in spec[s].items()}
                for s in sources}
    expected_items = {s: windowed_item_count(args.path, s, args.dst_start, comm.size,
                                             args.cache_size * comm.size)
                      for s in sources}

    gens = [NeuroH5ProjectionGen(args.path, s, "PopD", namespaces=["Synapses"],
                                 comm=comm, cache_size=args.cache_size)
            for s in sources]

    gen_local = {s: {} for s in sources}
    item_count = {s: 0 for s in sources}
    steps = 0
    for items in itertools.zip_longest(*gens):
        for s, item in zip(sources, items):
            if item is None:
                continue
            item_count[s] += 1
            gid, edges = item
            if gid is not None:
                gen_local[s][int(gid)] = edge_pairs(edges[0], edges[1]["Synapses"]["syn_id"])
        # One collective call per step, as the consumers of the generators
        # do; a rank that yields a different number of items deadlocks here.
        steps = comm.allreduce(steps + 1, op=MPI.MAX)

    graph, _ = scatter_read_graph(args.path, comm=comm, io_size=comm.size,
                                  projections=[(s, "PopD") for s in sources])
    scatter_local = {s: {int(g) for g, _ in graph["PopD"][s]} for s in sources}

    all_gen = comm.allgather({s: sorted(gen_local[s]) for s in sources})
    all_scatter = comm.allgather({s: sorted(scatter_local[s]) for s in sources})
    all_counts = comm.allgather(item_count)

    def home_ranks(per_rank, s):
        home = {}
        for rank, m in enumerate(per_rank):
            for g in m[s]:
                result.check(g not in home,
                             f"{s}: gid {g} delivered to ranks {home.get(g)} and {rank}")
                home[g] = rank
        return home

    gen_home = {s: home_ranks(all_gen, s) for s in sources}
    scatter_home = {s: home_ranks(all_scatter, s) for s in sources}

    for s in sources:
        result.check(set(gen_home[s]) == set(expected[s]),
                     f"{s}: generator gids {sorted(gen_home[s])} "
                     f"expected {sorted(expected[s])}")
        result.check(gen_home[s] == scatter_home[s],
                     f"{s}: generator gid->rank map differs from scatter_read_graph: "
                     + str({g: (r, scatter_home[s].get(g)) for g, r in gen_home[s].items()
                            if scatter_home[s].get(g) != r}))
        counts = [c[s] for c in all_counts]
        result.check(counts == [expected_items[s]] * comm.size,
                     f"{s}: item counts per rank {counts}, expected "
                     f"{expected_items[s]} on every rank")
        for g, pairs in gen_local[s].items():
            result.check(pairs == expected[s].get(g),
                         f"{s}: edges of gid {g}: got {pairs} expected {expected[s].get(g)}")

    split = {}
    for g in set(gen_home[sources[0]]).intersection(*[gen_home[s] for s in sources[1:]]):
        ranks = {gen_home[s][g] for s in sources}
        if len(ranks) > 1:
            split[g] = ranks
    result.check(not split, f"gids split across ranks between projections: {split}")

    result.info["item_count"] = item_count
    result.info["expected_item_count"] = expected_items
    result.info["local_gids"] = {s: len(gen_local[s]) for s in sources}

    result.finalize(args.out)


if __name__ == "__main__":
    main()
