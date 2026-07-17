import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKERS_DIR = Path(__file__).parent / "mpi_workers"


@pytest.fixture
def h5_path(tmp_path):
    return str(tmp_path / "test.h5")


def run_mpi_worker(worker_name, nranks, args, timeout=120):
    """Run mpi_workers/<worker_name>.py under `mpirun -n nranks`.

    The worker is expected to write a single JSON result document to the
    file path given via --out (see mpi_workers/_worker_common.py). Returns
    the parsed JSON dict. Raises AssertionError with full stdout/stderr on
    any non-zero exit (crash, MPI abort, etc.) so failures inside the
    worker are reported as an ordinary test failure rather than taking
    down the whole pytest session -- each MPI scenario runs in its own
    subprocess.
    """
    worker_path = WORKERS_DIR / f"{worker_name}.py"
    assert worker_path.exists(), f"no such worker script: {worker_path}"

    out_path = Path(args[args.index("--out") + 1]) if "--out" in args else None

    cmd = ["mpirun", "-n", str(nranks), sys.executable, str(worker_path), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    assert proc.returncode == 0, (
        f"mpi worker {worker_name} (n={nranks}) exited {proc.returncode}\n"
        f"cmd: {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    assert out_path is not None and out_path.exists(), (
        f"mpi worker {worker_name} did not produce a result file\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return json.loads(out_path.read_text())


@pytest.fixture
def mpi_worker(tmp_path):
    """Returns a callable: mpi_worker(worker_name, nranks, extra_args) -> result dict.

    Automatically appends a fresh --out <tmp_path>/result.json to extra_args.
    """
    def _run(worker_name, nranks, extra_args):
        out_path = tmp_path / f"{worker_name}_{nranks}.json"
        return run_mpi_worker(worker_name, nranks, [*extra_args, "--out", str(out_path)])
    return _run
