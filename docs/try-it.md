# Try it on your frame

This page is for Cal/Val users taking part in the v0.2 API review. It takes
about 30 minutes on a frame you know. What we want to learn: does the
everyday workflow do what you expect, are the names right, and what is
missing? Names can still change before v0.2; after it they are frozen.

## 1. Install

```bash
git clone https://github.com/mgovorcin/disp-xr.git
cd disp-xr
pixi install -e dev        # or: pip install -e ".[plot,aws]"
```

## 2. Run the workflow on your granules

Use a frame whose behaviour you know (a subsiding city, a stable area, a
GNSS site). A few ministacks are enough.

```python
import disp_xr as dxr

table = dxr.find_products("/path/to/your/granules")
print(table.groupby("reference_time").size())   # ministacks as you expect?

ds = dxr.open_disp(table, corrections=True)
ds = ds.disp.correct().disp.mask("recommended").disp.chain()
ds = ds.disp.rereference("point", lon=..., lat=...)   # a point you trust

ds.disp.plot_map()
ds.disp.plot_point(lon=..., lat=...)            # a place you know
dxr.to_zarr(ds, "my_frame.zarr")
```

Or the same from the command line:

```bash
disp_xr info /path/to/your/granules
disp_xr stack /path/to/your/granules --out my_frame.zarr --reference point --lon ... --lat ...
disp_xr export my_frame.zarr --format cog --out last.tif
```

## 3. Tell us

Open an issue on the repository (or reply to the review email) with:

1. **Frame and data**: frame id, number of granules, product version, where
   they were stored (local disk, S3).
2. **What worked and what did not**: errors, wrong-looking results, steps that
   were slow (with times), anything you had to look up.
3. **Names and defaults**: anything you would call differently. The open
   questions are listed in the [API review](api_review.md); a short answer to
   each helps.
4. **What is missing** for your Cal/Val work.

Thank you. Feedback is triaged into one issue per topic, and the agreed
changes are applied before v0.2 is tagged.
