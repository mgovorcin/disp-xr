# Test data

No real granules are stored in the repo. Tests build small synthetic DISP-S1
stacks at run time with `tests/fixtures/make_disp.py`, which copies the layout
recorded in [disp_s1_layout.md](disp_s1_layout.md) and returns the truth the
files were made from.

Ask for a stack by variant name through the session fixture `disp_stacks`:

```python
def test_something(disp_stacks):
    stack = disp_stacks["three_ministacks"]
    stack.paths            # the .nc files
    stack.cumulative       # displacement relative to the first date, (time, y, x)
```

Variants (`VARIANTS` in `make_disp.py`):

| Variant | What it holds |
| --- | --- |
| `two_ministacks` | 2 ministacks of 5 secondaries, one reference change |
| `three_ministacks` | 3 ministacks, two reference changes |
| `with_gaps` | `two_ministacks` without two secondaries of the first ministack |
| `duplicate` | `two_ministacks` plus an older-version copy of one granule |
| `mixed_dtype` | second ministack in the 2026 production style (float64 displacement, fixed time epoch) |
| `two_utm_zones` | two adjacent frames, UTM 10N and 11N |

Truth in every stack: a linear velocity gradient across x (-2 to +2 cm/yr), a
5 mm annual sinusoid, a 1 cm step at the middle epoch, and 1 mm Gaussian noise
(seeded). Quality features sit at fixed pixels (`LOW_COHERENCE`,
`UNWRAP_FAILED`, `WATER_COLUMNS`, `PS_PIXELS`, `CYCLE_PIXELS`), so tests can
assert exactly which pixels a mask removes.

A DISP-NISAR variant is added once the NISAR layout is recorded (plan task T5.2).
