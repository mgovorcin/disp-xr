# V&V of the linearity test and change points

Report: `docs/studies/series_vnv.md`. Geepers at the commit disp-xr pins, windows from the
smooth-trend validation (`GEEPERS_STUDY_CACHE`, read by `common.py`). Set
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`: each script runs one process per core.

    python noise_levels.py   # noise of real 500 m differences
    python synthetic.py      # results/synthetic.csv
    python fetch_windows.py f11116x <granules> <cache>/f38502v/stations.csv
    python real.py           # results/real.csv
    python offsets.py        # results/change_dates.csv and its ministacks

    # Houston
    python houston_stations.py
    python fetch_windows.py f08882 <granules> results/houston_stations.csv <enu.tif>
    python fetch_windows.py f38238 <granules> results/houston_stations.csv <enu.tif>
    python real.py real_houston.csv f08882 f08882t f38238
    python offsets.py change_dates_houston f08882t f38238
    python jump_f08882.py

    python end_to_end.py     # fit_series against the direct calls; needs disp-xr
    python summary.py > results/summary.txt
