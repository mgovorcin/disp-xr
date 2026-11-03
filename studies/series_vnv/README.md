# V&V of the linearity test and change points

Report: `docs/studies/series_vnv.md`. Geepers at the commit disp-xr pins, windows from the
smooth-trend validation (`GEEPERS_STUDY_CACHE`, read by `common.py`). Set
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`: each script runs one process per core.

    python noise_levels.py   # noise of real 500 m differences
    python synthetic.py      # results/synthetic.csv
    python fetch_overlap.py <cache>   # second track; needs h5py and pyproj
    python real.py           # results/real.csv
    python offsets.py        # results/change_dates.csv, ministacks.csv
    python end_to_end.py     # fit_series against the direct calls; needs disp-xr
    python summary.py > results/summary.txt
