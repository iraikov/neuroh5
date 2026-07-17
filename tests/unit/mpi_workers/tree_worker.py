#!/usr/bin/env python
"""MPI worker exercising neuroh5 scatter_read_trees / scatter_read_tree_selection.

Usage: mpirun -n N python tree_worker.py --scenario {scatter,scatter_selection}
           --path FILE --pop-start I --pop-count I --seed I --out FILE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker_common import WorkerResult, MPI  # noqa: E402
from _neuroh5_testing import make_trees, trees_equal  # noqa: E402

from neuroh5.io import scatter_read_trees, scatter_read_tree_selection  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["scatter", "scatter_selection"])
    p.add_argument("--path", required=True)
    p.add_argument("--pop-start", type=int, required=True)
    p.add_argument("--pop-count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    result = WorkerResult(comm)

    expected = make_trees(args.pop_start, args.pop_count, seed=args.seed)

    if args.scenario == "scatter":
        g, n_nodes = scatter_read_trees(args.path, "GC", comm=comm, io_size=1)
        got = dict(g)
        expected_local = {gid: v for gid, v in expected.items() if gid % comm.size == comm.rank}
        result.check(trees_equal(got, expected_local),
                     f"scatter mismatch: got {sorted(got.keys())} "
                     f"expected {sorted(expected_local.keys())}")
        result.check(n_nodes == args.pop_count, f"n_nodes {n_nodes} != {args.pop_count}")
        result.info["local_count"] = len(got)

    elif args.scenario == "scatter_selection":
        all_gids = sorted(expected.keys())
        local_selection = [g for i, g in enumerate(all_gids) if i % comm.size == comm.rank]
        sel, _ = scatter_read_tree_selection(args.path, "GC", local_selection,
                                             comm=comm, io_size=1)
        got = dict(sel)
        expected_local = {g: expected[g] for g in local_selection}
        result.check(trees_equal(got, expected_local),
                     f"scatter_selection mismatch: got {sorted(got.keys())} "
                     f"expected {sorted(expected_local.keys())}")
        result.info["local_selection_count"] = len(local_selection)

    result.finalize(args.out)


if __name__ == "__main__":
    main()
