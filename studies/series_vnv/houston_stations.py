"""Houston's GNSS stations with at least six years, from the T34 validation's list."""

from pathlib import Path

import pandas as pd
from geepers.gps_sources.unr import UnrSource

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    listed = pd.read_csv(
        HERE.parent / "gnss_validation/results/F08882_stations.csv", index_col=0
    )
    ids = listed.index[listed.gps_time_span_years >= 6]
    src = UnrSource()
    rows = [dict(id=s, lon=(ll := src.station_lonlat(s))[0], lat=ll[1]) for s in ids]
    pd.DataFrame(rows).to_csv(HERE / "results/houston_stations.csv", index=False)
    print(len(rows), "stations")
