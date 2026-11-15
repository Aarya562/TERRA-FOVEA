"""Stage 1: cheap geometric triage.

Runs on every point, costs a few milliseconds, and decides:
  * which returns are the ground surface, an obstacle, something overhead,
    or BELOW the expected surface (the signature of a pit or trench);
  * for ground returns, whether the surface is smooth and flat (drivable) or
    rough / steep (non-drivable), with an 'I cannot tell' outcome that is the
    only ground the neural network is asked about.

Ground model: a robust local plane per 1 m cell.
  1. lowest layer of returns in every 1 m cell;
  2. 5x5 median to throw away gross outliers (tree canopies, roofs);
  3. two rounds of re-weighted least-squares plane fitting in a 7 m window.
     Vehicles, kerbs and pits are a minority inside the window, fall outside
     the inlier band, and stop influencing the plane.
Because the plane ignores pits, returns from inside a pit come out BELOW the
plane.  That is how negative obstacles are found without any training data.
"""
import numpy as np
from scipy.ndimage import uniform_filter

from .config import Config
from . import classes as C
from .grid import KIND_GROUND, KIND_OBST, KIND_OVER, KIND_BELOW

_OFF = np.array([(i, j) for i in range(-2, 3) for j in range(-2, 3)])
_WIN = 7
_RIDGE = 3.0


def _plane_fit(zc, w, n):
    """Weighted LS plane z = a + b*(i-ic) + c*(j-jc) around every cell, via box-filter moments."""
    ii, jj = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")
    z = np.where(w > 0, zc, 0.0).astype(np.float64)
    N = _WIN * _WIN
    f = lambda a: uniform_filter(a, _WIN, mode="constant") * N
    S, Sx, Sy, Sz = f(w), f(w * ii), f(w * jj), f(w * z)
    Sxx, Sxy, Syy = f(w * ii * ii), f(w * ii * jj), f(w * jj * jj)
    Sxz, Syz = f(w * ii * z), f(w * jj * z)
    # shift moments to the cell centre
    mx, my = Sx - ii * S, Sy - jj * S
    mxx = Sxx - 2 * ii * Sx + ii * ii * S + _RIDGE
    myy = Syy - 2 * jj * Sy + jj * jj * S + _RIDGE
    mxy = Sxy - ii * Sy - jj * Sx + ii * jj * S
    mxz, myz = Sxz - ii * Sz, Syz - jj * Sz
    det = S * (mxx * myy - mxy * mxy) - mx * (mx * myy - mxy * my) + my * (mx * mxy - mxx * my)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (Sz * (mxx * myy - mxy * mxy) - mx * (mxz * myy - mxy * myz) + my * (mxz * mxy - mxx * myz)) / det
        b = (S * (mxz * myy - myz * mxy) - Sz * (mx * myy - mxy * my) + my * (mx * myz - mxz * my)) / det
        c = (S * (mxx * myz - mxy * mxz) - mx * (mx * myz - mxz * my) + Sz * (mx * mxy - mxx * my)) / det
    ok = (S >= 6) & (np.abs(det) > 1e-6)
    return a, b, c, ok, S


def _radial_reference(dx, dy, z, z0, max_range, n_sec=180, dr=1.0):
    """Coarse ground profile grown outward from the vehicle, sector by sector.

    Starting from the known ground under the sensor, each 1 m range bin is accepted as
    ground only if its lowest return continues the profile (within 0.2 m, plus 0.1 m per
    metre of unobserved gap).  Far away, ground returns are metres apart and a sign
    gantry or a roof can be the ONLY thing in a cell; without this gate such a cell
    would be mistaken for ground.  Returns the reference height per (sector, range bin).
    """
    nr = int(max_range / dr) + 2
    r = np.hypot(dx, dy)
    sec = ((np.arctan2(dy, dx) + np.pi) * (n_sec / (2 * np.pi))).astype(np.int64) % n_sec
    rb = np.minimum((r / dr).astype(np.int64), nr - 1)
    low = np.full(n_sec * nr, np.inf, np.float32)
    np.minimum.at(low, sec * nr + rb, z)
    low = low.reshape(n_sec, nr)
    low = np.minimum(low, np.minimum(np.roll(low, 1, 0), np.roll(low, -1, 0)))   # borrow from neighbour sectors
    ref = np.empty((n_sec, nr), np.float32)
    last = np.full(n_sec, z0, np.float32)
    gap = np.zeros(n_sec, np.float32)
    for k in range(nr):
        v = low[:, k]
        ok = np.isfinite(v) & (np.abs(v - last) < 0.2 + 0.1 * gap)
        last = np.where(ok, v, last)
        gap = np.where(ok, 0.0, gap + dr)
        ref[:, k] = last
    return ref, sec, rb


def segment(xw, yw, zw, sensor_xy, cfg: Config, sensor_z=None):
    cell = cfg.ground_cell
    n = int(2 * cfg.max_range / cell) + 4
    x0 = sensor_xy[0] - n * cell / 2; y0 = sensor_xy[1] - n * cell / 2
    fx = (xw - x0) / cell; fy = (yw - y0) / cell
    ix = np.clip(fx.astype(np.int64), 0, n - 1); iy = np.clip(fy.astype(np.int64), 0, n - 1)
    lin = ix * n + iy
    rng = np.hypot(xw - sensor_xy[0], yw - sensor_xy[1])

    # 1. lowest layer per cell
    zmin = np.full(n * n, np.inf, np.float32)
    np.minimum.at(zmin, lin, zw)
    low = zw < zmin[lin] + 0.10
    cnt_low = np.bincount(lin[low], minlength=n * n)
    zcell = np.full(n * n, np.nan)
    nz = cnt_low > 0
    zcell[nz] = np.bincount(lin[low], weights=zw[low], minlength=n * n)[nz] / cnt_low[nz]

    # 2. 5x5 median of the valid neighbours
    valid = np.flatnonzero(nz)
    pad = np.full((n + 4, n + 4), np.nan, np.float32)
    pad[2:-2, 2:-2] = zcell.reshape(n, n)
    vi, vj = valid // n, valid % n
    nb = pad[vi[:, None] + 2 + _OFF[:, 0], vj[:, None] + 2 + _OFF[:, 1]]
    nb.sort(axis=1)
    k = np.isfinite(nb).sum(1)
    med = np.full(n * n, np.nan)
    med[valid] = nb[np.arange(len(valid)), (k - 1) // 2]

    # 3. re-weighted plane fit
    zc2 = zcell.reshape(n, n); med2 = med.reshape(n, n)
    w = (np.abs(zc2 - med2) < 0.30).astype(np.float64)
    ref_c = None
    if sensor_z is not None:
        # radial gate: a cell can only support the ground model if it continues the profile from the vehicle
        ref, sec, rb = _radial_reference(xw - sensor_xy[0], yw - sensor_xy[1], zw, sensor_z - cfg.sensor_height, cfg.max_range)
        ci, cj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cdx = x0 + (ci + 0.5) * cell - sensor_xy[0]; cdy = y0 + (cj + 0.5) * cell - sensor_xy[1]
        csec = ((np.arctan2(cdy, cdx) + np.pi) * (ref.shape[0] / (2 * np.pi))).astype(np.int64) % ref.shape[0]
        crb = np.minimum(np.hypot(cdx, cdy).astype(np.int64), ref.shape[1] - 1)
        with np.errstate(invalid="ignore"):
            ref_c = ref[csec, crb].astype(np.float64)
            w *= (np.abs(zc2 - ref[csec, crb]) < 0.5)
            med2 = np.where(np.abs(med2 - ref[csec, crb]) < 0.6, med2, ref[csec, crb])
    a = med2.copy()
    b = np.zeros((n, n)); c = np.zeros((n, n))
    # Asymmetric bands: where two flat surfaces compete (road and a raised footpath 15 cm above
    # it), the ground is the LOWER one.  The first rounds therefore tolerate cells below the
    # current plane far more than cells above it, which walks the plane down onto the road;
    # the last round is tight on both sides so that real pits drop out again.
    for below, above in ((0.22, 0.05), (0.12, 0.06), (0.08, 0.08)):
        a_, b_, c_, ok, _ = _plane_fit(zc2, w, n)
        a = np.where(ok, a_, a); b = np.where(ok, b_, 0.0); c = np.where(ok, c_, 0.0)
        with np.errstate(invalid="ignore"):
            r = zc2 - a
            w = ((r > -below) & (r < above)).astype(np.float64)
    a_, b_, c_, ok, support = _plane_fit(zc2, w, n)
    a = np.where(ok, a_, a); b = np.where(ok, b_, 0.0); c = np.where(ok, c_, 0.0)
    af, bf, cf = a.reshape(-1), b.reshape(-1), c.reshape(-1)

    # height above the local plane, evaluated at the exact point position
    h = (zw - (af[lin] + bf[lin] * (fx - ix - 0.5) + cf[lin] * (fy - iy - 0.5))).astype(np.float32)
    tau = cfg.ground_tau0 + cfg.ground_tau_r * rng
    tneg = cfg.neg_tau0 + cfg.neg_tau_r * rng
    kind = np.full(xw.shape, KIND_OBST, np.int8)
    kind[h <= tau] = KIND_GROUND
    kind[h < -np.maximum(tneg, tau)] = KIND_BELOW
    kind[h >= cfg.clearance_split] = KIND_OVER
    kind[~np.isfinite(h)] = KIND_OBST

    # roughness (spread of ground returns about the plane) and slope, per cell
    isg = kind == KIND_GROUND
    lg = lin[isg]; hg = h[isg].astype(np.float64)
    cnt = np.bincount(lg, minlength=n * n).astype(np.float64)
    s1 = np.bincount(lg, weights=hg, minlength=n * n)
    s2 = np.bincount(lg, weights=hg * hg, minlength=n * n)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0))
    slope = np.degrees(np.arctan(np.hypot(bf, cf) / cell))

    gcls = np.full(xw.shape, -1, np.int8)
    c_std = std[lin]; c_cnt = cnt[lin]
    sure = isg & (c_cnt >= 12)
    steep = slope[lin] > cfg.slope_max_deg
    gcls[sure & (c_std < cfg.rough_std * 0.85) & ~steep] = C.ROAD
    gcls[sure & ((c_std > cfg.rough_std * 1.2) | steep)] = C.TERRAIN
    uncertain = isg & (gcls < 0)

    return {"h": h, "kind": kind, "gcls": gcls, "uncertain": uncertain, "range": rng.astype(np.float32),
            "plane": (a, b, c), "support": support, "origin": (x0, y0), "cell": cell, "n": n, "lin": lin,
            "cell_std": std.reshape(n, n), "cell_slope": slope.reshape(n, n)}


def ground_height(g, x, y):
    """Expected ground elevation at arbitrary world xy from the fitted planes."""
    n = g["n"]; cell = g["cell"]; x0, y0 = g["origin"]
    fx = (x - x0) / cell; fy = (y - y0) / cell
    ix = np.clip(fx.astype(np.int64), 0, n - 1); iy = np.clip(fy.astype(np.int64), 0, n - 1)
    a, b, c = g["plane"]
    return a[ix, iy] + b[ix, iy] * (fx - ix - 0.5) + c[ix, iy] * (fy - iy - 0.5)
