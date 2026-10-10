"""Station blocks of the T29 study stacks, for the T34 GNSS validation.

The study stacks hold one epoch per chunk, so reading a station's series
reads one chunk per epoch, and both tools under test (disp-xr and geepers)
would read every one for every station. This writes the 64 x 64 pixel
blocks around each GNSS station into a store on the same grid, chunked as
whole series: every pixel outside the blocks is missing, and a station's
series is one chunk. Each source chunk holding stations is read once, on a
process cluster, keeping only the blocks.

Run::

    pixi run -e test-py312 python extract.py F08882 F11116 F39362
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

HERE = Path(__file__).parent
STUDY = HERE.parent / "split_metric"
DATA = Path("/u/aurora-r0/govorcin/01_OPERA/DISP/data/split_metric")
LAYERS = ["displacement", "temporal_coherence", "phase_similarity"]
BLOCK = 64
# Pixels around a station that the sampling buffers reach (100 m is 4).
REACH = 8


def stack_path(frame: str) -> Path:
    """Return the unmasked study stack of ``frame``."""
    return DATA / frame / "unmasked" / "stack.zarr"


def store_path(frame: str) -> Path:
    """Return the station-block store of ``frame``."""
    return DATA / frame / "unmasked" / "stations.zarr"


def station_blocks(ds: xr.Dataset, stations: pd.DataFrame) -> set[tuple[int, int]]:
    """Return the (row, col) block indices that hold any station's reach."""
    rows = np.abs(ds.y.values[:, None] - stations.y.values).argmin(axis=0)
    cols = np.abs(ds.x.values[:, None] - stations.x.values).argmin(axis=0)
    blocks = set()
    for r, c in zip(rows, cols):
        for dr in (-REACH, REACH):
            for dc in (-REACH, REACH):
                rr = min(max(r + dr, 0), ds.sizes["y"] - 1)
                cc = min(max(c + dc, 0), ds.sizes["x"] - 1)
                blocks.add((rr // BLOCK, cc // BLOCK))
    return blocks


def read_blocks(frame: str, blocks: list[tuple[int, int]]) -> dict:
    """Read the series of ``blocks`` (all epochs, all layers) into memory.

    Only the blocks are kept, so a worker holds a few MB rather than the
    whole source chunk the decoder reads them from.
    """
    ds = xr.open_zarr(stack_path(frame))[LAYERS]
    out = {}
    for br, bc in blocks:
        window = ds.isel(
            y=slice(br * BLOCK, (br + 1) * BLOCK), x=slice(bc * BLOCK, (bc + 1) * BLOCK)
        )
        out[br, bc] = {name: window[name].values for name in LAYERS}
    return out


def extract(frame: str, client) -> Path:
    """Write the station-block store of ``frame`` and return its path."""
    ds = xr.open_zarr(stack_path(frame))[LAYERS]
    stations = pd.read_csv(STUDY / "stations" / f"{frame}.csv")
    blocks = station_blocks(ds, stations)
    size = ds.displacement.encoding["chunks"][-1]  # the source read chunk
    assert size % BLOCK == 0, size
    per = size // BLOCK
    parents: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for b in blocks:
        parents.setdefault((b[0] // per, b[1] // per), []).append(b)

    out = store_path(frame)
    template = ds.drop_encoding()
    template = template.chunk({"time": -1, "y": BLOCK, "x": BLOCK})
    template = xr.full_like(template, np.nan)
    encoding = {name: {"write_empty_chunks": False} for name in LAYERS}
    template.to_zarr(out, mode="w", compute=False, encoding=encoding, zarr_format=3)
    group = zarr.open_group(out, mode="r+")
    print(
        f"{frame}: {len(stations)} stations, {len(blocks)} blocks in "
        f"{len(parents)} source chunks",
        flush=True,
    )

    from distributed import as_completed

    futures = {
        client.submit(read_blocks, frame, members, pure=False): parent
        for parent, members in parents.items()
    }
    for done, future in enumerate(as_completed(futures), start=1):
        for (br, bc), data in future.result().items():
            for name in LAYERS:
                block = data[name]
                group[name][
                    :,
                    br * BLOCK : br * BLOCK + block.shape[1],
                    bc * BLOCK : bc * BLOCK + block.shape[2],
                ] = block
        print(f"  {done}/{len(parents)} source chunks", flush=True)
    return out


def main(frames: list[str]) -> None:
    """Extract the station blocks of each frame on a process cluster."""
    from distributed import Client, LocalCluster

    # Processes: zarr decodes chunks in one thread per process.
    with (
        LocalCluster(n_workers=16, threads_per_worker=1, memory_limit="8GB") as cluster,
        Client(cluster) as client,
    ):
        for frame in frames:
            print(extract(frame, client), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["F08882", "F11116", "F39362"])
