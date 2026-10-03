# Third-party notices

disp-xr does not depend on [opera-utils](https://github.com/opera-adt/opera-utils).
The functions it needs from that project are ported into disp-xr and maintained
here (see `docs/porting.md`). opera-utils is dual-licensed BSD-3-Clause or
Apache-2.0; disp-xr uses it under BSD-3-Clause. The license text is in
[`LICENSES/opera-utils-BSD-3-Clause.txt`](LICENSES/opera-utils-BSD-3-Clause.txt).

Copyright (c) 2022 California Institute of Technology ("Caltech"). U.S. Government
sponsorship acknowledged.

Every ported file carries a header naming its source. Ported code, by disp-xr file:

| disp-xr file | Ported names | opera-utils source | Source commit |
| --- | --- | --- | --- |
| `src/disp_xr/search.py` | `UrlType`, `from_filename`, `_get_download_url`, `from_umm`, `search` | `src/opera_utils/disp/_search.py`, `_product.py`, `constants.py` | Not recorded (copied before 2025-06) |
| `src/disp_xr/product.py` | `DispGranule` filename parsing, deduplication in `find_products` | `src/opera_utils/disp/_product.py` (`DispProduct`, `DispProductStack`), `constants.py` (`DISP_FILE_REGEX`) | `dff4b45` |
| `src/disp_xr/_chunks.py` | `clamp_chunks` | `src/opera_utils/disp/_utils.py` (`_clamp_chunk_dict`) | `dff4b45` |
| `src/disp_xr/_credentials.py` | `ASFEndpoint`, `AWSCredentials`, `get_temporary_aws_credentials`, `get_earthdata_username_password` | `src/opera_utils/credentials.py` | `dff4b45` |
| `src/disp_xr/_remote.py` | `is_remote_url`, `get_filesystem`, `open_file`, `open_h5` | `src/opera_utils/_remote.py` | `dff4b45` |
| `src/disp_xr/export/mintpy.py` | MintPy file set and metadata keys | `src/opera_utils/disp/mintpy.py` | `dff4b45` |
| `src/disp_xr/export/zarr.py` | `round_mantissa`, per-variable encoding | `src/opera_utils/disp/_utils.py`, `_reformat.py` | `dff4b45` |
| `src/disp_xr/geometry/static.py` | CSLC-STATIC layer names and nodata values | `src/opera_utils/geometry.py` | `dff4b45` |
| `src/disp_xr/geometry/los.py` | Slant-range triangle (`slant_range`) | `src/opera_utils/geometry.py` | `dff4b45` |
| `src/disp_xr/quality.py` | `combine_quality_masks` | `src/opera_utils/disp/_reformat.py` | `dff4b45` |
| `src/disp_xr/reference.py` | `ReferenceMethod`, `get_reference_values`, center-of-mass pick in `find_stable_point` | `src/opera_utils/disp/_enums.py`, `_reference.py`, `rebase_reference.py` | `dff4b45` |
| `src/disp_xr/stack.py` | `NaNPolicy` and the ministack-chaining approach (rewritten: offsets taken from the crossover epoch by date) | `src/opera_utils/disp/_rebase.py` | `dff4b45` |

New ports name the opera-utils commit they were taken from (`dff4b45` for the
ports planned in `docs/porting.md`). Add a row here in the same change as the port.

## geepers

disp-xr does not depend on [geepers](https://github.com/opera-adt/geepers). Its
colored-noise trend estimator is ported into disp-xr (the PRD decided the S1
estimator lives here). geepers is licensed under the Apache License 2.0; the
license text is in [`LICENSES/geepers-Apache-2.0.txt`](LICENSES/geepers-Apache-2.0.txt)
and its NOTICE, which Section 4(d) requires with any derivative work, in
[`LICENSES/geepers-NOTICE.txt`](LICENSES/geepers-NOTICE.txt).

Copyright (c) 2025-2026 California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.

The ported file keeps the geepers SPDX header and names its source and changes.

| disp-xr file | Ported names | geepers source | Source commit |
| --- | --- | --- | --- |
| `src/disp_xr/timeseries/noise.py` | `_fractional_diff_coeffs`, `_powerlaw_covariance`, `_softmax_weights`, `_nll_core`, the PLWN, FNWN, RWFNWN and WN likelihood searches, the Whittle power-law estimate, the final GLS solve of `estimate_trend` | `src/geepers/trend.py` | `d542178` |

`tests/data/noise_reference.json` holds geepers results on synthetic series,
written by `tests/data/make_noise_reference.py` in a geepers environment.
