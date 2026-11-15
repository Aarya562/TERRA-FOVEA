"""Proofs that changing resolution never loses or misplaces safety-relevant data."""
import numpy as np
from terrafovea import morton, classes as C
from terrafovea.config import Config
from terrafovea.grid import AdaptiveGrid, coarsen_block, refine_block, KIND_OBST, KIND_GROUND, KIND_OVER, K
from terrafovea.pipeline import cell_state
from terrafovea.metrics import paint


def _random_block(r, m=5, ncell=256):
    f = {"zg": r.normal(0, 0.2, (m, ncell)).astype(np.float32), "ng": r.integers(0, 50, (m, ncell)).astype(np.uint16),
         "zt": r.uniform(0.2, 2, (m, ncell)).astype(np.float32), "zc": r.uniform(2.5, 6, (m, ncell)).astype(np.float32),
         "lo": r.integers(0, 12, (m, ncell)).astype(np.int8), "dyn": r.integers(0, 2, (m, ncell)).astype(np.uint8) * 255,
         "neg": r.integers(0, 255, (m, ncell)).astype(np.uint8), "stamp": r.integers(0, 900, (m, ncell)).astype(np.uint16),
         "hist": r.integers(0, 90, (m, ncell, K)).astype(np.uint8)}
    f["zg"][f["ng"] == 0] = np.nan
    f["zt"][r.random((m, ncell)) < 0.6] = np.nan
    f["lo"][~np.isfinite(f["zt"])] = 0
    f["zc"][r.random((m, ncell)) < 0.8] = np.nan
    return f


def test_z_order_parent_is_a_bit_shift():
    lut, ix, iy = morton.tables(32)
    for L in range(1, 5):
        lutL, _, _ = morton.tables(32 >> L)
        yy, xx = np.mgrid[0:32, 0:32]
        assert np.array_equal(lut[yy, xx] >> (2 * L), lutL[yy >> L, xx >> L])
    assert np.array_equal(lut[iy, ix], np.arange(32 * 32))


def test_refine_then_coarsen_is_identity():
    f = _random_block(np.random.default_rng(1))
    back = coarsen_block(refine_block(f, lo_tentative=127))
    for k in f:
        assert np.array_equal(back[k], f[k], equal_nan=True), k


def test_coarsen_keeps_everything_that_matters():
    f = _random_block(np.random.default_rng(2))
    c = coarsen_block(f)
    q = lambda a: a.reshape(a.shape[0], -1, 4)
    zt = q(f["zt"]); zc = q(f["zc"])
    assert np.array_equal(np.isfinite(c["zt"]), np.isfinite(zt).any(2))                    # no obstacle vanishes
    assert np.allclose(c["zt"][np.isfinite(c["zt"])], np.nanmax(zt, 2)[np.isfinite(c["zt"])])   # tallest height kept
    assert np.allclose(c["zc"][np.isfinite(c["zc"])], np.nanmin(zc, 2)[np.isfinite(c["zc"])])   # lowest ceiling kept
    assert np.array_equal(c["lo"] > 0, (q(f["lo"]) > 0).any(2))                             # occupied stays occupied
    assert np.array_equal(c["neg"], q(f["neg"]).max(2))                                     # pit confidence kept
    assert np.array_equal(c["hist"], f["hist"].reshape(f["hist"].shape[0], -1, 4, K).max(2))    # no class out-voted
    zg = q(f["zg"]); lo_, hi_ = np.nanmin(zg, 2), np.nanmax(zg, 2)
    ok = np.isfinite(c["zg"])
    assert np.all(c["zg"][ok] >= lo_[ok] - 1e-5) and np.all(c["zg"][ok] <= hi_[ok] + 1e-5)   # ground stays in range


def _filled_grid(policy_levels, seed=3):
    cfg = Config()
    g = AdaptiveGrid(cfg)
    g.set_window(0.0, 0.0)
    r = np.random.default_rng(seed)
    n = 60000
    x = r.uniform(-60, 60, n); y = r.uniform(-60, 60, n)
    kind = np.where(r.random(n) < 0.25, KIND_OBST, KIND_GROUND).astype(np.int8)
    z = np.where(kind == KIND_OBST, r.uniform(0.3, 2.0, n), r.normal(0, 0.01, n)).astype(np.float32)
    cls = np.where(kind == KIND_OBST, r.integers(2, 8, n), 0).astype(np.int8)
    ts = cfg.tile_size
    cx = (g.ox + np.arange(g.Wn) + 0.5) * ts; cy = (g.oy + np.arange(g.Wn) + 0.5) * ts
    d = np.hypot(*np.meshgrid(cx, cy, indexing="ij"))
    lm = policy_levels(d).astype(np.int8)
    g.insert(x, y, z, kind, cls, 1, lm)
    return cfg, g, (x, y, z, kind)


def test_every_point_lands_in_a_cell_that_contains_it():
    cfg, g, (x, y, z, kind) = _filled_grid(lambda d: np.searchsorted([10, 20, 40], d))
    ok, tid, lvl, flat = g.locate(x, y)
    assert ok.all()
    for L in g.levels:
        m = lvl == L
        if not m.any():
            continue
        n = g.T >> L; ncell = n * n
        cx, cy = g.cell_xy(L, tid[m], flat[m] % ncell)
        res = cfg.level_res(L)
        assert np.all(np.abs(cx - x[m]) <= res / 2 + 1e-9) and np.all(np.abs(cy - y[m]) <= res / 2 + 1e-9)
    # all four levels really are in use, so the band boundaries are exercised
    assert len(np.unique(lvl)) == 4


def test_no_obstacle_is_lost_at_resolution_boundaries():
    cfg, g, (x, y, z, kind) = _filled_grid(lambda d: np.searchsorted([10, 20, 40], d))
    ok, tid, lvl, flat = g.locate(x, y)
    o = kind == KIND_OBST
    for L in g.levels:
        m = o & (lvl == L)
        P = g.pools[L].a
        assert np.all(P["lo"].reshape(-1)[flat[m]] > 0)
        assert np.all(P["zt"].reshape(-1)[flat[m]] >= z[m] - 1e-6)       # stored top is never below a real return


def test_live_regrid_never_uncovers_an_obstacle():
    cfg, g, _ = _filled_grid(lambda d: np.zeros_like(d))
    x0 = y0 = -62.0; n = int(124 / cfg.base_res)
    def blocked():
        c = g.cells(); return paint(c, cell_state(c, cfg)["blocked"], x0, y0, n, n, cfg)
    before = blocked()
    ids = g.tile_ids()
    r = np.random.default_rng(5)
    g.regrid(ids, r.integers(1, 5, len(ids)).astype(np.int8))           # coarsen every tile by a random amount
    mid = blocked()
    assert not (before & ~mid).any()
    g.regrid(ids, np.zeros(len(ids), np.int8))                           # and back to the finest level
    after = blocked()
    assert not (before & ~after).any()
    assert np.array_equal(mid, after)                                    # refining invents nothing new


def test_memory_accounting_matches_live_cells():
    cfg, g, _ = _filled_grid(lambda d: np.searchsorted([10, 20, 40], d))
    tiles, cells = g.level_counts()
    assert g.n_cells() == cells.sum()
    assert g.memory_bytes() == cells.sum() * 27 + tiles.sum() * 16


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
