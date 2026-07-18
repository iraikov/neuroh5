import json
import os
import signal
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

    A hang (MPI collective deadlock) is reported as a distinct, explicit
    TIMED OUT AssertionError rather than letting subprocess.run raise an
    uncaught TimeoutExpired: that alternative produces a bare "process did
    not return exit code 0"-style failure with none of the diagnostic
    stdout/stderr this function otherwise attaches, making a hang
    indistinguishable from a crash in the pytest report.
    """
    worker_path = WORKERS_DIR / f"{worker_name}.py"
    assert worker_path.exists(), f"no such worker script: {worker_path}"

    out_path = Path(args[args.index("--out") + 1]) if "--out" in args else None

    # Enable Open MPI oversubscription
    mpirun_extra_args = ["--oversubscribe"] if os.environ.get("NEUROH5_MPIRUN_OVERSUBSCRIBE") else []
    cmd = ["mpirun", *mpirun_extra_args, "-n", str(nranks), sys.executable, str(worker_path), *args]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        raise AssertionError(
            f"mpi worker {worker_name} (n={nranks}) TIMED OUT after {timeout}s "
            "-- this is a hang (likely an MPI collective deadlock), not a crash\n"
            f"cmd: {' '.join(cmd)}\n--- stdout so far ---\n{stdout}\n"
            f"--- stderr so far ---\n{stderr}"
        )

    if proc.returncode != 0:
        result_json = None
        if out_path is not None and out_path.exists():
            try:
                result_json = out_path.read_text()
            except OSError:
                pass
        raise AssertionError(
            f"mpi worker {worker_name} (n={nranks}) exited {proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
            f"--- result json ({out_path}) ---\n{result_json}"
        )

    assert out_path is not None and out_path.exists(), (
        f"mpi worker {worker_name} did not produce a result file\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
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
