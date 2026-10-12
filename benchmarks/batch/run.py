"""Record the instance, then run ``bench_downsample.py`` (plan task T28.5).

Prints ``BENCH_INFO`` (instance type, CPU, memory, load, page-fault speed)
before the benchmark and ``BENCH_DONE`` (exit code, page-fault speed, load)
after it, so a run on a noisy instance can be told from a clean one.
"""

import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import disp_xr

METADATA = "http://169.254.169.254/latest"


def instance_type() -> str:
    """Return the EC2 instance type from the metadata service (IMDSv2)."""
    try:
        token_request = urllib.request.Request(
            f"{METADATA}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_request, timeout=2).read().decode()
        request = urllib.request.Request(
            f"{METADATA}/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(request, timeout=2).read().decode()
    except OSError as err:  # the container may not reach the metadata service
        return f"unavailable ({type(err).__name__})"


def touch_seconds() -> float:
    """Return the time to allocate and touch 256 MB (slow on a busy host)."""
    start = time.perf_counter()
    block = np.ones(2**28 // 8)
    block[::512] = 2
    del block
    return round(time.perf_counter() - start, 3)


def cpu_model() -> str:
    """Return the CPU model name."""
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> int:
    """Run the benchmark between the two records; return its exit code."""
    pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    info = {
        "instance_type": instance_type(),
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "mem_gb": round(pages / 2**30),
        "load": os.getloadavg(),
        "python": platform.python_version(),
        "disp_xr": disp_xr.__version__,
        "touch_256mb_s": [touch_seconds() for _ in range(3)],
    }
    print("BENCH_INFO " + json.dumps(info), flush=True)
    # The job command names the script and its arguments; by default the
    # downsample benchmark. A CSV written to /tmp/out.csv is printed after.
    args = sys.argv[1:] or ["bench_downsample.py"]
    rc = subprocess.call([sys.executable, f"/app/{args[0]}", *args[1:]])
    out = Path("/tmp/out.csv")
    if out.exists():
        print("BENCH_CSV_BEGIN\n" + out.read_text() + "BENCH_CSV_END", flush=True)
    done = {"rc": rc, "touch_after_s": touch_seconds(), "load": os.getloadavg()}
    print("BENCH_DONE " + json.dumps(done), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
