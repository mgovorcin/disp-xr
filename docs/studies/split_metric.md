# Split metric and weight defaults (T29)

Status (2026-10-04): frames, inputs, method and runner done (T29.1 to
T29.5); runs in progress; results and the decision (T29.6) to come.

This study picks the default quadtree split metric (`displacement`,
`velocity` or `structure`, T28) and the T26 weight parameters (`penalty`,
`f0`, floors, exponents, and the minimum residual weight of T26.6). It
compares them at 1, 2 and 10 km cells on four criteria:

1. reconstruction error against the 30 m field (velocity and displacement),
2. GNSS agreement per cell (T32),
3. gradient preservation across known features,
4. cell count and runtime.

Scripts and inputs live in `studies/split_metric/` at the repository root.

## Frames

Three DISP-S1 frames, one per kind of signal. Details (paths, bounds,
dates, product version and why each was chosen) are in
`studies/split_metric/frames.yaml`.

| Frame | Role | Site | Granules | Dates | GNSS used / inside |
| --- | --- | --- | --- | --- | --- |
| F08882 | subsidence | Houston, upper Texas Gulf Coast | 269 | 2016-09 to 2026-02 | 97 / 144 |
| F11116 | tectonic, topography-correlated troposphere | Central California: creeping San Andreas, San Joaquin Valley | 389 | 2016-07 to 2026-04 | 126 / 151 |
| F39362 | quiet (troposphere control) | Canberra and the Snowy Mountains | 247 | 2016-07 to 2025-04 | 25 / 34 |

- F11116 spans the November 2025 production change: 75 of its 389
  granules store displacement as float64, the rest as float32. They are read
  together (see the product notes).
- F08882 is also the cal-disp frame, so GNSS results can be compared with
  that work.
- F11116 has a sharp velocity step at the creeping fault trace, the
  gradient-preservation test case for a tectonic signal. With the Coast
  Ranges and Sierra foothills it is also the deforming site with
  topography-correlated troposphere that T31.7 asked for.
- F39362 is the control: little deformation but about 1500 m of relief from
  the coast to the Snowy Mountains, so troposphere that follows elevation
  shows up without a deformation signal under it. Its granules are product
  version 0.10 (a stakeholder run outside North America), not 1.0.

New Orleans (F44055) was considered and dropped: it is a second subsidence
site and has no GNSS prepared.

### GNSS stations

`studies/split_metric/stations.py` writes `stations/<frame>.csv`: the UNR
stations inside each frame's grid over the frame's dates, with first and
last day, number of daily solutions and the share of the DISP span covered.
A station is used (`use = True`) when it covers at least half of the span
and has at least 2.5 years of data.

`gnss_los.py` projects each used station's daily east, north and up onto
the line of sight at its pixel and fits a MIDAS velocity over the frame's
dates (`stations/<frame>_los.csv`). LOS vectors come from DISP-S1-STATIC
for F08882 and F11116. Canberra has no DISP-S1-STATIC, so its LOS layers
come from the stakeholder run, built from CSLC-S1-STATIC bursts on the same
grid as the granules. Six Canberra stations fall outside the valid LOS
coverage, which leaves 19.

### DISP-NISAR frames for T35.5

Frames of the Whirlwind NISAR campaign, which already has GNSS for each:

| NISAR frame | Pairs with | GNSS stations |
| --- | --- | --- |
| 042_D_070 | F11116 (same area) | 232 |
| 062_A_016 | F08882 (same area) | 100 |
| 133_A_022 | F39362 (no campaign frame covers Australia; a quiet site with moderate relief in the US Northeast instead) | 77 |

## Inputs

`prepare.py` builds one study stack per frame, following `disp_xr stack`:

- product corrections;
- the recommended mask;
- chaining of the ministacks;
- a reference.

It keeps the displacement and the three quality layers and every third
epoch: 91 epochs for F08882, 84 for F39362, 130 for F11116. Cumulative
displacement stays valid on a subset of epochs. It also writes an OLS
velocity, which is the 30 m reference for the velocity criteria and the
input of the velocity split metric.

Two choices differ from the command line, both for run time on full
frames. The study only compares methods with each other, so neither
changes its results.

- **Reference.** The median of a 2 km box around a GNSS station near the
  frame center (UHEB, TURO, ...) instead of `auto`, whose stable-point
  search reads every quality layer of every epoch first.
- **Floors.** The weight floors default to the product's own quality
  thresholds (`similarity_threshold`, `temporal_coherence_threshold` on
  `recommended_mask`), as `quality_weights` does. These differ between
  granules and frames, so the most common values are recorded on each
  stack: 0.6 and 0.7 for F08882, 0.5 and 0.6 for F39362.

## Design: equal point budgets

A metric that splits more always reconstructs better, so the metrics are
compared at equal point counts:

- **Regular grid.** For each cell size (1, 2 and 10 km, factors 33, 67
  and 333), `downsample` gives the regular grid. Its number of cells with
  data is the point budget.
- **Quadtrees.** Each split metric then builds a quadtree with the same
  budget (`quadtree(n_points=...)`, which picks the threshold whose leaf
  count is closest). Levels run from 3 (240 m) to 10 (30 km).

The regular grid is the baseline: a quadtree metric is worth its cost only
where it beats the grid at the same number of points.

## Criteria

Each partition (grid or quadtree) is painted back on the 30 m grid through
its leaf-id raster: every pixel takes the value of its cell. Then:

1. **Reconstruction.** The RMSE against the 30 m field over reliable pixels
   (static weight above zero, data at some epoch). For displacement it is
   the RMS over epochs, and also given for the last epoch. For velocity,
   each cell's velocity is the weighted mean of the 30 m velocity.
2. **GNSS agreement.** The cell velocity at each station minus its LOS
   velocity, as an RMS after removing a plane in x and y. DISP is relative
   to its reference and GNSS is absolute, and plate motion on a line of
   sight that changes across the frame is a ramp. The `full` row gives
   the same number for the 30 m pixel at each station, the floor the
   partitions are measured against.
3. **Gradient preservation.** Along straight profiles across known features
   (`profiles` in `frames.yaml`), a 1 km swath is binned every 300 m. The
   criterion is the RMSE between the binned 30 m velocity profile and the
   painted one. The profiles are:
   - three 30 km profiles across the creeping San Andreas (F11116);
   - one across the northwest Harris County subsidence (F08882);
   - one from the coast up to the Snowy Mountains (F39362).
4. **Cost.** The number of points and the seconds to build the partition.

## Weight sweep

`run.py --sweep` varies the quality-weight parameters one at a time around
the defaults of `quality_weights`:

- `penalty` 0.25, 0.75, 1 (no penalty);
- `f0` 0.1, 0.5;
- `min_residual` 0.01, 0.2;
- the similarity floor 0.2, 0.5;
- the coherence floor 0.5, 0.8;
- without the residual layer;
- equal weights.

It runs at 2 km on the grid and on the frame's best split metric (by the
sum of its ranks on criteria 1 to 3). The weights change cell means much
more than which metric wins, and the sweep is the costly part of the
study.

## Running it

```bash
cd studies/split_metric
pixi run -e test-py312 python stations.py      # station lists
pixi run -e test-py312 python gnss_los.py      # GNSS LOS velocities
pixi run -e test-py312 python prepare.py F08882 F39362 F11116
pixi run -e test-py312 python run.py F08882 F39362 F11116
pixi run -e test-py312 python run.py --sweep F08882 F39362 F11116
```

`prepare.py` and `run.py` run dask on 24 local processes. Granule reads
hold the HDF5 lock, and zarr decodes in one thread per process, so threads
ran at about one core. The stacks are about 22 GB each.

## Found along the way

The study turned up five problems in disp-xr, each fixed with a test:

- quadtree cells on the frame edge never split;
- functions that align chunks failed on stacks read back from Zarr;
- `quality_weights` loaded the whole residual stack into memory;
- `rereference("area")` read the whole frame for a small box;
- `open_disp` refused (and could not chain) frames that span the November
  2025 production change. The product notes have the details.

## Results

To come: one table per frame (`results/<frame>.csv`, `results/<frame>_sweep.csv`).

## Decision

To come (T29.6): the default split metric and weight parameters, set in
code, and the PRD S2 open questions closed.
