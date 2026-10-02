"""Quick-look plots of a DISP frame: a displacement map and a point time series.

Run on a directory of DISP granules (needs the ``plot`` extra)::

    python docs/gallery/quicklook.py /path/to/granules --out quicklook

It chains the ministacks, re-references to a stable point, and writes
``map.png`` (last epoch) and ``point.png`` (the scene center, median over
90 m).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import disp_xr as dxr  # noqa: E402


def main() -> None:
    """Write the two quick-look figures for the granules given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("granules", type=Path)
    parser.add_argument("--out", type=Path, default=Path("quicklook"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = dxr.open_disp(args.granules).disp.mask("recommended").disp.stack()
    ds = ds.disp.rereference("auto")

    ax = ds.disp.plot_map()  # last epoch, in mm, symmetric scale
    ax.figure.savefig(args.out / "map.png", dpi=150)

    row, col = ds.sizes["y"] // 2, ds.sizes["x"] // 2
    ax = ds.disp.plot_point(row=row, col=col, buffer_m=90)
    ax.figure.savefig(args.out / "point.png", dpi=150)
    print(f"Wrote {args.out / 'map.png'} and {args.out / 'point.png'}")


if __name__ == "__main__":
    main()
