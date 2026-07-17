"""Common utilities for mpi_workers/*.py scripts.

Each worker is a standalone script run once per rank under `mpirun`. It
computes a local pass/fail plus diagnostics, combines pass/fail across all
ranks with an MPI allreduce (so a failure on any single rank fails the
whole scenario and is visible from every rank's perspective, not just
rank 0's), and then has rank 0 write a single JSON result file for the
parent pytest process to inspect.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI  # noqa: E402


class WorkerResult:
    def __init__(self, comm):
        self.comm = comm
        self.ok = True
        self.errors = []
        self.info = {}

    def check(self, condition, message):
        if not condition:
            self.ok = False
            self.errors.append(f"rank {self.comm.rank}: {message}")

    def finalize(self, out_path):
        import json

        all_ok = self.comm.allreduce(self.ok, op=MPI.LAND)
        gathered_errors = self.comm.gather(self.errors, root=0)
        gathered_info = self.comm.gather(self.info, root=0)

        if self.comm.rank == 0:
            errors = [e for sub in gathered_errors for e in sub]
            Path(out_path).write_text(json.dumps({
                "ok": all_ok,
                "errors": errors,
                "info_by_rank": gathered_info,
                "nranks": self.comm.size,
            }))

        self.comm.barrier()
        # Every rank exits nonzero if the scenario failed, so a crash-free
        # but logically-wrong result still shows up as a subprocess failure.
        if not all_ok:
            sys.exit(1)
