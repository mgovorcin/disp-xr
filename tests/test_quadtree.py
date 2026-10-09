"""Quadtree downsampling (plan task T28)."""

import numpy as np
import pytest
import xarray as xr

import disp_xr as dxr
from disp_xr.quadtree import quadtree


def _dataset(nt=4, ny=45, nx=70, seed=0, chunks=None):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Smooth west, a sharp bowl in the east, so the tree splits there.
    bowl = 0.02 * np.exp(-((xx - 0.75 * nx) ** 2 + (yy - ny / 2) ** 2) / 30.0)
    t = np.arange(nt)[:, None, None]
    values = t * bowl + 0.001 * rng.standard_normal((nt, ny, nx))
    values[:, :6, :10] = np.nan  # a hole: no reliable pixels there
    weights = rng.random((nt, ny, nx))
    x = 500000.0 + 30.0 * np.arange(nx) + 15
    y = 4100000.0 - 30.0 * np.arange(ny) - 15
    ds = xr.Dataset(
        {"displacement": (("time", "y", "x"), values, {"units": "meters"})},
        coords={"time": np.arange(nt), "y": y, "x": x},
    ).rio.write_crs("EPSG:32610")
    w = xr.DataArray(weights, dims=("time", "y", "x"), coords=ds.displacement.coords)
    if chunks:
        ds, w = ds.chunk(chunks), w.chunk(chunks)
    return ds, w


class _Reference:
    """Slow recursive quadtree straight from the pixels."""

    def __init__(self, ds, w, threshold, metric, min_level, max_level, min_reliable):
        self.v = ds.displacement.values
        self.w = w.values
        self.static = self.w.mean(axis=0)
        self.reliable = (self.static > 0) & np.isfinite(self.v).any(axis=0)
        self.args = threshold, metric, min_level, max_level, min_reliable
        self.leaves = {}

    def block(self, k, i, j):
        s = 2**k
        return slice(i * s, (i + 1) * s), slice(j * s, (j + 1) * s)

    def moments(self, k, i, j):
        rs, cs = self.block(k, i, j)
        means, variances, sigma2 = [], [], []
        for t in range(self.v.shape[0]):
            v, w = self.v[t, rs, cs].ravel(), self.w[t, rs, cs].ravel()
            ok = np.isfinite(v)
            v, w = v[ok], w[ok]
            if w.sum() <= 0:
                means.append(np.nan), variances.append(np.nan), sigma2.append(np.nan)
                continue
            m = np.sum(w * v) / w.sum()
            var = np.sum(w * (v - m) ** 2) / w.sum()
            means.append(m), variances.append(var)
            sigma2.append(var / (w.sum() ** 2 / np.sum(w**2)))
        return np.array(means), np.array(variances), np.array(sigma2)

    def score(self, k, i, j):
        threshold, metric, min_level, *_ = self.args
        if metric == "displacement":
            return np.sqrt(np.nanmean(self.moments(k, i, j)[1]))
        if k == min_level:
            return np.nan
        kids = [
            self.moments(k - 1, 2 * i + a, 2 * j + b) for a in (0, 1) for b in (0, 1)
        ]
        means = np.array([m for m, _, _ in kids])  # (4, time)
        noise = np.array([s for _, _, s in kids])
        spread = np.nanvar(means, axis=0)
        return np.sqrt(np.nanmean(spread)) / np.sqrt(
            np.nanmean(np.nanmean(noise, axis=0))
        )

    def n_reliable(self, k, i, j):
        rs, cs = self.block(k, i, j)
        return int(self.reliable[rs, cs].sum())

    def n_inside(self, k, i, j):
        rs, cs = self.block(k, i, j)
        return self.reliable[rs, cs].size

    def visit(self, k, i, j):
        threshold, _, min_level, _, min_reliable = self.args
        if k > min_level:
            kids = [(2 * i + a, 2 * j + b) for a in (0, 1) for b in (0, 1)]
            enough = all(
                self.n_reliable(k - 1, a, b)
                >= min_reliable * self.n_inside(k - 1, a, b)
                for a, b in kids
            )
            with np.errstate(invalid="ignore"):
                if enough and self.score(k, i, j) > threshold:
                    for a, b in kids:
                        self.visit(k - 1, a, b)
                    return
        if self.n_reliable(k, i, j) > 0:
            self.leaves[(k, i, j)] = self.moments(k, i, j)[0]

    def run(self):
        *_, max_level, _ = self.args
        ny, nx = self.reliable.shape
        s = 2**max_level
        for i in range(-(-ny // s)):
            for j in range(-(-nx // s)):
                self.visit(max_level, i, j)
        return self.leaves


@pytest.mark.parametrize(
    ("metric", "threshold"), [("displacement", 0.004), ("structure", 2.0)]
)
def test_matches_recursive_reference(metric, threshold):
    ds, w = _dataset(chunks={"y": 16, "x": 24})
    tree = quadtree(
        ds, threshold=threshold, metric=metric, min_level=1, max_level=4, weights=w
    )
    ref = _Reference(ds, w, threshold, metric, 1, 4, 0.5).run()
    points = tree.points.compute()
    got = {
        (int(k), int(i), int(j)): n
        for n, (k, i, j) in enumerate(
            zip(points.level.values, points.row.values, points.col.values)
        )
    }
    assert set(got) == set(ref)
    assert len({k for k, _, _ in ref}) > 1  # the tree really split
    for key, n in got.items():
        np.testing.assert_allclose(
            points.displacement.values[:, n], ref[key], rtol=1e-10, equal_nan=True
        )
    # The leaf-id raster paints each leaf's pixels with its index.
    for (k, i, j), n in got.items():
        s = 2**k
        block = tree.leaf_id.values[i * s : (i + 1) * s, j * s : (j + 1) * s]
        assert (block == n).all()
    assert tree.leaf_id.shape == (45, 70)


def test_hole_and_counts():
    ds, w = _dataset()
    tree = quadtree(
        ds, threshold=0.004, metric="displacement", min_level=1, max_level=4, weights=w
    )
    points = tree.points.compute()
    assert (points.valid_fraction <= 1).all()
    np.testing.assert_allclose(points.size, 30.0 * 2**points.level)
    # Every pixel with data is in a leaf; centers sit inside their cells.
    covered = tree.leaf_id.values >= 0
    has_data = np.isfinite(ds.displacement.values).any(axis=0)
    assert covered[has_data].all()
    half = points.size / 2
    assert (abs(points.centroid_x - points.x) <= half).all()
    assert points.attrs["downsample_grid"] == "quadtree"
    raster = tree.to_raster("displacement", time=-1)
    leaf = tree.leaf_id.values
    np.testing.assert_array_equal(
        raster.values[leaf >= 0],
        points.displacement.isel(time=-1).values[leaf[leaf >= 0]],
    )
    assert np.isnan(raster.values[leaf < 0]).all()
    assert "quadtree" in points.attrs["history"]


def test_velocity_and_custom_metric():
    ds, w = _dataset()
    velocity = ds.displacement.isel(time=-1) / 3.0
    tree = quadtree(
        ds,
        threshold=0.001,
        metric="velocity",
        velocity=velocity,
        min_level=1,
        max_level=4,
        weights=w,
    )
    assert tree.points.level.min() < 4  # splits around the bowl
    calls = []

    def never(level, children):
        calls.append(children is None)
        return xr.zeros_like(level["sum_w"].isel(time=0, drop=True))

    flat = quadtree(
        ds, threshold=0.5, metric=never, min_level=1, max_level=3, weights=w
    )
    assert set(flat.points.level.values) == {3}
    assert calls.count(True) == 1  # min_level has no children


@pytest.mark.parametrize("metric", ["displacement", "structure"])
def test_n_points_picks_the_closest_threshold(metric):
    ds, w = _dataset()
    args = {"metric": metric, "min_level": 1, "max_level": 4, "weights": w}
    counts = {}
    for target in (40, 120, 400):
        tree = quadtree(ds, n_points=target, **args)
        n = tree.points.sizes["point"]
        threshold = tree.points.attrs["quadtree_threshold"]
        # Same leaves as asking for that threshold directly.
        again = quadtree(ds, threshold=threshold, **args)
        np.testing.assert_array_equal(again.leaf_id.values, tree.leaf_id.values)
        # No other score value as threshold gets closer to the target.
        for nearby in (np.nextafter(threshold, -np.inf), threshold * 1.05):
            if np.isfinite(nearby):
                other = quadtree(ds, threshold=nearby, **args).points.sizes["point"]
                assert abs(n - target) <= abs(other - target)
        counts[target] = n
    assert counts[40] <= counts[120] <= counts[400]


def test_n_points_out_of_range():
    ds, w = _dataset()
    args = {"min_level": 1, "max_level": 4, "weights": w}
    coarse = quadtree(ds, n_points=1, **args)
    assert set(coarse.points.level.values) == {4}
    fine = quadtree(ds, n_points=10**6, **args)
    finest = quadtree(ds, threshold=-1.0, **args)
    assert fine.points.sizes["point"] == finest.points.sizes["point"]


@pytest.mark.parametrize("metric", ["displacement", "structure"])
def test_edge_cells_split(metric):
    # Cells larger than the frame, or hanging over its edge, split like the
    # others: children outside the frame do not block the split.
    ds, w = _dataset(ny=45, nx=70)
    # No hole, so only the frame edge could block a split. Noise, not a
    # constant: `structure` does not split cells with no noise estimate.
    noise = np.random.default_rng(0).normal(0, 1e-3, ds.displacement.shape)
    ds = ds.fillna(xr.zeros_like(ds.displacement) + noise)
    tree = quadtree(
        ds, threshold=-1.0, metric=metric, min_level=1, max_level=7, weights=w
    )
    assert set(tree.points.level.values) == {1}


def test_errors():
    ds, w = _dataset()
    with pytest.raises(ValueError, match="velocity="):
        quadtree(ds, threshold=1.0, metric="velocity", weights=w)
    with pytest.raises(ValueError, match="min_level"):
        quadtree(ds, threshold=1.0, min_level=4, max_level=2, weights=w)
    with pytest.raises(ValueError, match="not both"):
        quadtree(ds, weights=w)
    with pytest.raises(ValueError, match="not both"):
        quadtree(ds, threshold=1.0, n_points=10, weights=w)
    with pytest.raises(ValueError, match="positive"):
        quadtree(ds, n_points=0, weights=w)
    with pytest.raises(ValueError, match="unknown metric"):
        quadtree(ds, threshold=1.0, metric="curvature", weights=w)


def test_quality_weights_on_fixture(stack_dir):
    ds = dxr.stack_disp(dxr.open_disp(stack_dir)).disp.rereference("median")
    tree = quadtree(ds, threshold=0.003, min_level=1, max_level=4)
    points = tree.points.compute()
    assert points.sizes["point"] > 0
    assert np.isfinite(points.displacement.isel(time=-1)).any()
    assert dxr.quadtree is quadtree
