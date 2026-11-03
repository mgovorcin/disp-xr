"""F08882 in autumn 2019: when the offsets fall, and the frame-wide step at the stations."""

import warnings

import pandas as pd
from common import NAME, RESULTS, folder_of, pixel_series

warnings.simplefilter("ignore")


def main():
    c = pd.read_csv(RESULTS / "change_dates_houston.csv", parse_dates=["date"])
    c = c[(c.role == "pixel") & (c.offset.abs() > 15) & (c.frame == "f08882t")]
    window = c.date.between("2019-07-01", "2020-02-01")
    print(f"offsets >15 mm in 2019-07 to 2020-01: {window.sum()} of {len(c)}")
    print(c[window].date.dt.to_period("M").value_counts().sort_index().to_string())
    for frame in ("f08882", "f38238"):
        folder = folder_of(frame)
        refs = sorted(
            {
                pd.Timestamp(NAME.search(p.name).group(1))
                for p in folder.glob("OPERA*.npz")
            }
        )
        print(
            frame,
            "ministack starts 2019-2020:",
            [str(r.date()) for r in refs if r.year in (2019, 2020)],
        )
        dates, cube = pixel_series(folder)
        s = pd.DataFrame(cube[:, :, 0], index=dates)
        s = s.loc["2019-06":"2020-03"]
        step = s.diff()
        print(
            pd.DataFrame(
                {
                    "station median": s.median(axis=1),
                    "median step": step.median(axis=1),
                    "spread of step": step.std(axis=1),
                    "stations": s.notna().sum(axis=1),
                }
            )
            .round(1)
            .to_string()
        )


if __name__ == "__main__":
    main()
