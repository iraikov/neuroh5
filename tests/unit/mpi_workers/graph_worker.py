#!/usr/bin/env python
"""MPI worker exercising neuroh5 graph scatter/bcast/selection reads, and
multi-rank append_graph.

Usage: mpirun -n N python graph_worker.py
           --scenario {scatter,bcast,scatter_selection,append}
           --path FILE --dst-start I --dst-count I --src-start I --src-count I
           --seed I [--with-attrs] [--io-size I] --out FILE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker_common import WorkerResult, MPI  # noqa: E402
from _neuroh5_testing import make_edges, edges_equal, normalize_edge_attrs  # noqa: E402

from neuroh5.io import (  # noqa: E402
    scatter_read_graph,
    bcast_graph,
    scatter_read_graph_selection,
    append_graph,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                    choices=["scatter", "bcast", "scatter_selection", "append"])
    p.add_argument("--path", required=True)
    p.add_argument("--dst-start", type=int, required=True)
    p.add_argument("--dst-count", type=int, required=True)
    p.add_argument("--src-start", type=int, required=True)
    p.add_argument("--src-count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-attrs", action="store_true")
    p.add_argument("--io-size", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    result = WorkerResult(comm)

    expected = make_edges(args.dst_start, args.dst_count, args.src_start, args.src_count,
                          seed=args.seed, with_attrs=args.with_attrs)
    namespaces = ["Synapses"] if args.with_attrs else []

    def normalize(got, attr_info):
        if not args.with_attrs:
            return got
        name_index = attr_info["PopB"]["PopA"]["Synapses"]
        return normalize_edge_attrs(got, name_index)

    if args.scenario == "scatter":
        g, attr_info = scatter_read_graph(args.path, comm=comm, namespaces=namespaces, io_size=1)
        got = normalize(dict(g["PopB"]["PopA"]), attr_info)
        total_nodes = args.src_count + args.dst_count
        expected_local = {
            dst: expected[dst] for dst in expected
            if dst % comm.size == comm.rank and dst < total_nodes
        }
        result.check(edges_equal(got, expected_local, with_attrs=args.with_attrs),
                     f"scatter mismatch: got {sorted(got.keys())} "
                     f"expected {sorted(expected_local.keys())}")
        result.info["local_dst_count"] = len(got)
        result.info["local_edge_count"] = sum(len(v[0]) for v in got.values())

    elif args.scenario == "bcast":
        g, attr_info = bcast_graph(args.path, comm=comm, namespaces=namespaces)
        got = normalize(dict(g["PopB"]["PopA"]), attr_info)
        result.check(edges_equal(got, expected, with_attrs=args.with_attrs),
                     f"bcast mismatch: got {len(got)} dsts expected {len(expected)}")
        result.info["dst_count"] = len(got)

    elif args.scenario == "scatter_selection":
        selection = sorted(expected.keys())
        sel, attr_info = scatter_read_graph_selection(args.path, selection, comm=comm,
                                                       namespaces=namespaces, io_size=1)
        got = normalize(dict(sel["PopB"]["PopA"]), attr_info)
        result.check(edges_equal(got, expected, with_attrs=args.with_attrs),
                     f"scatter_selection mismatch: got {len(got)} expected {len(expected)}")

    elif args.scenario == "append":
        # Regression test for a real deadlock/data-loss bug: append_graph
        # used to hang (or silently drop non-root ranks' contributions)
        # whenever more than one rank acted as an I/O rank
        # (io_size > 1), because dst_blk_idx/dst_blk_ptr/dst_ptr were each
        # independently extended and collectively written by every I/O
        # rank -- multiple ranks colllectively writing disjoint hyperslabs
        # of what's typically a single small HDF5 chunk reliably deadlocked
        # inside HDF5's parallel I/O. Fixed by gathering those three (small)
        # datasets to the root I/O rank and having it write them alone;
        # src_idx (large) keeps its original distributed collective write.
        #
        # Each rank contributes a disjoint round-robin slice -- the same
        # partitioning append_graph's internal node_rank_map applies
        # regardless of how the caller splits the data -- and results are
        # verified via scatter_read_graph (merged across ranks with
        # allgather), not plain read_graph, since read_graph itself splits
        # its result across ranks by block range rather than returning the
        # full graph on every rank.
        local_edges = {
            gid: v for i, (gid, v) in enumerate(sorted(expected.items()))
            if i % comm.size == comm.rank
        }
        io_kwargs = {} if args.io_size == 0 else {"io_size": args.io_size}
        append_graph(args.path, {"PopB": {"PopA": local_edges}}, comm=comm, **io_kwargs)
        comm.barrier()

        g, attr_info = scatter_read_graph(args.path, comm=comm, namespaces=namespaces, io_size=1)
        got_local = normalize(dict(g["PopB"]["PopA"]), attr_info)
        merged = {}
        for d in comm.allgather(got_local):
            merged.update(d)
        result.check(edges_equal(merged, expected, with_attrs=args.with_attrs),
                     f"append mismatch: got {sorted(merged.keys())} "
                     f"expected {sorted(expected.keys())}")
        result.info["local_contributed"] = len(local_edges)
        result.info["merged_total"] = len(merged)

    result.finalize(args.out)


if __name__ == "__main__":
    main()
