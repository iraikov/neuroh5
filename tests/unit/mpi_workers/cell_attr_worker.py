#!/usr/bin/env python
"""MPI worker exercising neuroh5 cell attribute scatter/bcast/selection reads.

Usage: mpirun -n N python cell_attr_worker.py --scenario {scatter,bcast,scatter_selection}
           --path FILE --pop-start I --pop-count I --seed I --out FILE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker_common import WorkerResult, MPI  # noqa: E402
from _neuroh5_testing import make_cell_attrs, cell_attrs_equal  # noqa: E402

from neuroh5.io import (  # noqa: E402
    scatter_read_cell_attributes,
    scatter_read_cell_attribute_selection,
    bcast_cell_attributes,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                    choices=["scatter", "bcast", "scatter_selection"])
    p.add_argument("--path", required=True)
    p.add_argument("--pop-start", type=int, required=True)
    p.add_argument("--pop-count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    result = WorkerResult(comm)

    expected = make_cell_attrs(args.pop_start, args.pop_count, seed=args.seed)

    if args.scenario == "scatter":
        d = scatter_read_cell_attributes(args.path, "GC", namespaces=["Test"],
                                         comm=comm, io_size=1)
        got = dict(d["Test"])
        # Default round-robin: rank = gid % size (over the whole node index space).
        expected_local = {gid: v for gid, v in expected.items() if gid % comm.size == comm.rank}
        result.check(cell_attrs_equal(got, expected_local),
                     f"scatter mismatch: got {sorted(got.keys())} "
                     f"expected {sorted(expected_local.keys())}")
        result.info["local_count"] = len(got)

    elif args.scenario == "bcast":
        got = dict(bcast_cell_attributes(args.path, "GC", 0, namespace="Test", comm=comm))
        result.check(cell_attrs_equal(got, expected),
                     f"bcast mismatch: got {len(got)} expected {len(expected)}")
        result.info["count"] = len(got)

    elif args.scenario == "scatter_selection":
        # Each rank asks for a distinct, deliberately uneven slice of gids
        # to confirm the selection is honored per-rank, not just broadcast.
        all_gids = sorted(expected.keys())
        local_selection = [g for i, g in enumerate(all_gids) if i % comm.size == comm.rank]
        got = dict(scatter_read_cell_attribute_selection(
            args.path, "GC", local_selection, namespace="Test", comm=comm, io_size=1))
        expected_local = {g: expected[g] for g in local_selection}
        result.check(cell_attrs_equal(got, expected_local),
                     f"scatter_selection mismatch: got {sorted(got.keys())} "
                     f"expected {sorted(expected_local.keys())}")
        result.info["local_selection_count"] = len(local_selection)

    result.finalize(args.out)


if __name__ == "__main__":
    main()
