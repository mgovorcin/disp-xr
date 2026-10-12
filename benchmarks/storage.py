"""Compression and chunk layout of DISP layers in Zarr v3 (plan task T37.3b, local).

Writes a window of a real chained frame (displacement and temporal
coherence) and, with ``--granules``, the integer layers of the granules
(masks, connected components, SHP counts) to local Zarr v3 stores with each
codec and chunk layout, and records:

- the compression ratio (uncompressed bytes over stored bytes), which does
  not depend on the machine;
- the write time, and the read time of two DISP access patterns: one map
  (one epoch, the whole window) and one pixel's time series.

Trials are interleaved across configurations and every row records the time
to touch fresh memory (a stalling node makes timings meaningless; see
docs/performance.md). The S3 half of T37.3b (NetCDF on S3, virtual Icechunk,
native Zarr on S3) is ``storage_s3.py``.

Run::

    pixi run -e test python benchmarks/storage.py STACK.zarr --out results.csv
    pixi run -e test python benchmarks/storage.py --granules DIR --out ints.csv
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

import disp_xr as dxr
from disp_xr.export.zarr import round_mantissa

CODECS = {
    "zstd-1": [ZstdCodec(level=1)],
    "zstd-3 (default)": [ZstdCodec(level=3)],
    "zstd-9": [ZstdCodec(level=9)],
    "blosc-zstd-3 shuffle": [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
    "blosc-zstd-3 bitshuffle": [
        BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    ],
    "blosc-lz4-5 shuffle": [BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")],
}
# (time, y, x) chunks inside 1-epoch or whole-series shards, as the presets.
LAYOUTS = {
    "maps 1x1024x1024": (1, 1024, 1024),
    "maps 1x256x256": (1, 256, 256),
    "timeseries Tx128x128": (-1, 128, 128),
}
INTEGER_LAYERS = (
    "recommended_mask",
    "persistent_scatterer_mask",
    "water_mask",
    "connected_component_labels",
    "shp_counts",
)


def page_touch_seconds(nbytes: int = 2**28) -> float:
    """Return the time to allocate and touch ``nbytes`` of fresh memory."""
    start = time.perf_counter()
    block = np.ones(nbytes // 8)
    elapsed = time.perf_counter() - start
    del block
    return elapsed


def store_bytes(path: Path) -> int:
    """Return the bytes stored under ``path``."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def write(values: np.ndarray, path: Path, chunks: tuple, compressors: list) -> float:
    """Write ``values`` as one Zarr v3 array and return the seconds taken."""
    chunks = tuple(values.shape[0] if c == -1 else c for c in chunks)
    start = time.perf_counter()
    array = zarr.create_array(
        store=str(path),
        shape=values.shape,
        chunks=chunks,
        dtype=values.dtype,
        compressors=compressors,
        fill_value=np.nan if values.dtype.kind == "f" else 0,
        overwrite=True,
    )
    array[:] = values
    return time.perf_counter() - start


def read_times(path: Path) -> tuple[float, float]:
    """Return the seconds to read one map and one pixel's series."""
    array = zarr.open_array(str(path), mode="r")
    start = time.perf_counter()
    array[array.shape[0] // 2]
    one_map = time.perf_counter() - start
    start = time.perf_counter()
    array[:, array.shape[1] // 2, array.shape[2] // 2]
    series = time.perf_counter() - start
    return one_map, series


def load_window(stack: Path, epochs: int, size: int) -> dict[str, np.ndarray]:
    """Return a centered window of the stack's displacement and coherence."""
    ds = xr.open_zarr(stack)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    window = {
        "time": slice(1, 1 + epochs),  # skip the zero epoch
        "y": slice(ny // 2 - size // 2, ny // 2 + size // 2),
        "x": slice(nx // 2 - size // 2, nx // 2 + size // 2),
    }
    return {
        name: ds[name].isel(window).values.astype(np.float32)
        for name in ("displacement", "temporal_coherence")
        if name in ds
    }


def load_integer_window(granules: Path, epochs: int, size: int) -> dict:
    """Return a centered window of the granules' integer layers, as stored."""
    ds = dxr.open_disp(granules, layers=list(INTEGER_LAYERS))
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    window = {
        "time": slice(0, epochs),
        "y": slice(ny // 2 - size // 2, ny // 2 + size // 2),
        "x": slice(nx // 2 - size // 2, nx // 2 + size // 2),
    }
    return {name: ds[name].isel(window).values for name in INTEGER_LAYERS}


def run(
    stack: Path | None,
    epochs: int,
    size: int,
    trials: int,
    granules: Path | None = None,
) -> pd.DataFrame:
    """Benchmark every layer, codec and layout; return one row per trial."""
    layers = {}
    if stack is not None:
        layers = load_window(stack, epochs, size)
        # Lossy option, for comparison only: 12 mantissa bits keep about 3
        # significant digits (10 micrometers on a centimeter).
        layers["displacement (12 bits kept)"] = round_mantissa(
            layers["displacement"], keep_bits=12
        )
    if granules is not None:
        layers |= load_integer_window(granules, epochs, size)
    work = Path(tempfile.mkdtemp())
    rows = []
    try:
        for trial in range(trials):
            touch = page_touch_seconds()
            for layer, values in layers.items():
                for codec, compressors in CODECS.items():
                    for layout, chunks in LAYOUTS.items():
                        path = work / "a.zarr"
                        seconds = write(values, path, chunks, compressors)
                        stored = store_bytes(path)
                        one_map, series = read_times(path)
                        rows.append(
                            {
                                "layer": layer,
                                "codec": codec,
                                "layout": layout,
                                "trial": trial,
                                "ratio": values.nbytes / stored,
                                "stored_mb": stored / 1e6,
                                "write_s": seconds,
                                "read_map_s": one_map,
                                "read_series_s": series,
                                "page_touch_s": touch,
                                "dtype": str(values.dtype),
                            }
                        )
                        shutil.rmtree(path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return pd.DataFrame(rows)


def main() -> None:
    """Parse arguments, run, print the median per configuration."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stack", type=Path, nargs="?", help="a chained stack written by to_zarr"
    )
    parser.add_argument(
        "--granules", type=Path, help="DISP granules, for the integer layers"
    )
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stack is None and args.granules is None:
        parser.error("give a stack, --granules or both")
    table = run(args.stack, args.epochs, args.size, args.trials, args.granules)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = table.groupby(["layer", "codec", "layout"], sort=False).median(
        numeric_only=True
    )
    columns = ["ratio", "write_s", "read_map_s", "read_series_s"]
    print(summary[columns].round(3).to_string())
    print(f"page touch (256 MB): max {table['page_touch_s'].max():.2f} s")


if __name__ == "__main__":
    main()
