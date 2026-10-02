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
| `src/disp_xr/quality.py` | `combine_quality_masks` | `src/opera_utils/disp/_reformat.py` | `dff4b45` |
| `src/disp_xr/stack.py` | `NaNPolicy` and the ministack-chaining approach (rewritten: offsets taken from the crossover epoch by date) | `src/opera_utils/disp/_rebase.py` | `dff4b45` |

New ports name the opera-utils commit they were taken from (`dff4b45` for the
ports planned in `docs/porting.md`). Add a row here in the same change as the port.
