# DISP-S1 granule layout

Recorded on 2026-10-01 from real granules with h5py and xarray (plan task T2.2). The synthetic fixtures in `tests/fixtures/make_disp.py` copy this layout.

| Granule | Frame | Production |
| --- | --- | --- |
| `OPERA_L3_DISP-S1_IW_F44055_VV_20160714T000153Z_20160807T000155Z_v1.0_20250412T164531Z.nc` | 44055, UTM 16N, 7721 x 9462 | 2025 |
| `OPERA_L3_DISP-S1_IW_F11116_VV_20191105T140736Z_20191123T140818Z_v1.0_20260920T055848Z.nc` | 11116, UTM 10N, 6898 x 9618 | 2026 |

## Filename

`OPERA_L3_DISP-S1_IW_F<frame:5>_VV_<reference>Z_<secondary>Z_v<version>_<production>Z.nc`, datetimes as `%Y%m%dT%H%M%S`.

## Groups

| Group | Contents |
| --- | --- |
| `/` | 2D layers `(y, x)`, coordinates `x`, `y`, `time`, the `reference_time` variable, `spatial_ref` |
| `/corrections` | `ionospheric_delay`, `solid_earth_tide`, `perpendicular_baseline` `(y, x)`; scalars `reference_point`, `spatial_ref`; its own `time`, `x`, `y` |
| `/identification` | 52 scalar datasets, mostly fixed-length byte strings (`frame_id`, `product_version`, `reference_datetime`, `secondary_datetime`, `radar_wavelength` = 0.05546576, `acquisition_mode` = `IW`, ...) |
| `/metadata` | 59 scalar datasets; large fixed-length strings: `pge_runconfig` (`|S50862` in one 2025 granule), `dolphin_workflow_config` (`|S64571` in the same granule) |

## Data layers

Every 2D layer has dims `(y, x)`: there is no time dimension on the layers in the file. Chunks are 256 x 256, gzip level 4, with shuffle, in both productions. `grid_mapping = "spatial_ref"` on every layer.

| Layer | dtype | `_FillValue` | units | Values |
| --- | --- | --- | --- | --- |
| `displacement` | **float32 (2025), float64 (2026)** | NaN | meters | Relative to the granule's reference date |
| `short_wavelength_displacement` | float32 | NaN | meters | |
| `temporal_coherence` | float32 | NaN | unitless | 0 to 1 |
| `phase_similarity` | float32 | NaN | unitless | 0 to 1 |
| `estimated_phase_quality` | float32 | NaN | unitless | |
| `timeseries_inversion_residuals` | float32 | NaN | radians | |
| `recommended_mask` | uint8 | 255 | unitless | 1 good, 0 bad. Attributes `temporal_coherence_threshold = "0.6"`, `similarity_threshold = "0.3"` |
| `water_mask` | uint8 | 255 | unitless | 1 land, 0 water |
| `persistent_scatterer_mask` | uint8 | 255 | unitless | 1 PS |
| `connected_component_labels` | uint16 | 65535 | unitless | 0 means unwrapping failed |
| `shp_counts` | int16 | 0 | unitless | |

## Coordinates and time

- `x`, `y` are float64 pixel centers (`metadata/product_pixel_coordinate_convention = "pixel center"`), spacing 30 m, `y` decreasing. `spatial_ref/GeoTransform` gives the corner: x0 = x[0] - 15, y0 = y[0] + 15.
- `spatial_ref` is an int64 scalar with CF grid-mapping attributes, including `crs_wkt` and `GeoTransform`.
- `time` is a length-1 dimension coordinate. `reference_time` is a length-1 variable on `time`, not a dimension scale.
- **2025 production:** `time.units = "seconds since <secondary datetime>"`, value 0; `reference_time.units = "seconds since <reference datetime>"`, value 0. The epoch differs in every file.
- **2026 production:** both use `"seconds since 2010-01-01 00:00:00.000000"` with nonzero values.
- xarray (`engine="h5netcdf"`) decodes both to the right datetimes. Sizes `{time: 1, y, x}`, coords `x`, `y`, `time`; `reference_time` is a data variable.

## Ministacks

In frame F44055 (209 granules, 14 ministacks), each ministack holds 14 or 15 secondaries, and the next ministack's reference date equals the last secondary date of the previous ministack.

## DISP-NISAR (pre-release)

Recorded on 2026-10-02 from the SAS golden outputs in
`03_DISP_NI/Releases/delivery_data_small/golden_output/historical` (frame 26410,
version 0.4, 11112 x 11328).

- Same layers, dtypes, chunking (256 x 256, gzip 4, shuffle), coordinates and
  time encoding as the 2025 DISP-S1 production, except
  `corrections/ionospheric_delay` is float64.
- Root `title = "OPERA_L3_DISP-NISAR Product"`.
- Extra group `science/LSAR/identification` (`lookDirection`,
  `orbitPassDirection`, `trackNumber`).
- File names are `<reference YYYYMMDD>_<secondary YYYYMMDD>.nc`: the SAS output,
  before the delivery wrapper renames it. No official OPERA DISP-NISAR name has
  been seen.
- `/identification` still carries Sentinel-1 placeholder values in this release
  (`radar_band = C`, `acquisition_mode = IW`, `radar_wavelength = 0.2419...`), so
  only `frame_id`, `product_version` and the two datetimes are used.

## DISP-S1-STATIC

Recorded on 2026-10-02 from `OPERA_L3_DISP-S1-STATIC_F11116_20140403_S1A_v1.0`
(CMR collection `OPERA_L3_DISP-S1-STATIC_V1`, 1445 frames at that date).

- One granule per frame, three GeoTIFFs named
  `OPERA_L3_DISP-S1-STATIC_F<frame:5>_<validity YYYYMMDD>_<platform>_v<version>_<layer>.tif`.
- Same CRS, shape and transform as the frame's DISP granules (F11116: EPSG
  32610, 6898 x 9618, corner 499800, 4114500, 30 m). Deflate, 128 x 128 blocks
  (256 x 256 for the mask). Tags carry `FRAME_ID`, `ORBIT_DIRECTION`,
  `TRACK_NUMBER`, `PRODUCT_VERSION`.

| File | Bands, dtype | nodata | Values |
| --- | --- | --- | --- |
| `line_of_sight_enu` | 3, float32 | 0 | East, north, up of the ground-to-satellite LOS unit vector (F11116, descending: east 0.50 to 0.71, up 0.69 to 0.86) |
| `dem` | 1, float32 | none | Copernicus GLO-30, metres (tag `dem_egm_model = EGM2008`) |
| `layover_shadow_mask` | 1, uint8 | 255 | 0 not masked, 1 shadow, 2 layover, 3 both |

No water mask: the DISP granules carry `water_mask`. No DISP-NISAR static
collection exists in CMR yet.

## Not recorded yet

- An official DISP-NISAR granule.
- Other DISP-S1 product versions.
