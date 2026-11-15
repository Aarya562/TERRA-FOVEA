"""Range-image visibility: one test, two uses.

The current sweep is rendered into a small range image (azimuth x elevation,
nearest return per pixel).  Any 3D point in the map can then be asked:
"did the laser travel THROUGH you this sweep?"  If the measured range in that
direction is clearly longer than the distance to the point, yes.

  * Ghost removal.  A cell that the map believes is occupied, but which the
    laser has just passed through at obstacle height, is no longer there.
    Cells that belonged to moving things are cleared on the first see-through;
    static cells need several.  The test is done at the stored obstacle height,
    so a beam that passes ABOVE a low kerb does not erase the kerb.  This is the
    same physics as Bresenham ray tracing, but it costs one lookup per occupied
    cell instead of a walk along every one of ~100 000 rays.

  * Occlusion awareness.  Cells hidden behind something are left alone:
    'not seen' is never treated as 'free'.
"""
import numpy as np
from scipy.ndimage import minimum_filter

from .config import Config
from . import classes as C


class RangeImage:
    def __init__(self, pts_sensor, el_top_deg=None, el_bot_deg=None, rows=56, cols=720):
        x, y, z = pts_sensor[:, 0], pts_sensor[:, 1], pts_sensor[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        el = np.arctan2(z, np.hypot(x, y))
        if el_top_deg is None:
            hi, lo = np.percentile(el, [99.9, 0.1])
        else:
            hi, lo = np.deg2rad(el_top_deg), np.deg2rad(el_bot_deg)
        self.hi, self.lo, self.rows, self.cols = float(hi), float(lo), rows, cols
        row, col, ok = self._pix(x, y, z)
        img = np.full(rows * cols, np.inf, np.float32)
        np.minimum.at(img, (row * cols + col)[ok], r[ok].astype(np.float32))
        self.img = img.reshape(rows, cols)
        # widen along azimuth only: the beam below always meets the ground sooner, so widening
        # in elevation would make every distant cell look occluded
        self.dil = minimum_filter(self.img, size=(1, 3), mode=("nearest", "wrap"))

    def _pix(self, x, y, z):
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.hypot(x, y))
        col = ((az + np.pi) * (self.cols / (2 * np.pi))).astype(np.int64) % self.cols
        rowf = (self.hi - el) * (self.rows / (self.hi - self.lo))
        ok = (rowf >= 0) & (rowf < self.rows)
        row = np.clip(rowf.astype(np.int64), 0, self.rows - 1)
        return row, col, ok

    def seen_through(self, x, y, z_lo, z_hi, margin, allow_no_return=None):
        """True where the laser demonstrably passed through the vertical segment [z_lo, z_hi]
        standing at sensor-frame (x, y).

        Only image rows that look entirely INTO the segment are consulted.  Near the horizon one
        row spans tens of metres of ground, so a row that merely grazes the bottom of the segment
        would report the ground in front of it and make every distant cell look occluded.

        A direction with no return at all (open road to the horizon) proves nothing for a static
        cell - the surface might just be dark.  For cells that held a moving object
        (`allow_no_return`) it is accepted as free: the object was reflective a moment ago.
        """
        rho = np.hypot(x, y)
        r = np.sqrt(rho * rho + z_hi * z_hi)
        az = np.arctan2(y, x)
        col = ((az + np.pi) * (self.cols / (2 * np.pi))).astype(np.int64) % self.cols
        k = self.rows / (self.hi - self.lo)
        row_top = (self.hi - np.arctan2(z_hi, rho)) * k
        row_bot = (self.hi - np.arctan2(z_lo, rho)) * k
        r0 = np.ceil(row_top).astype(np.int64); r1 = np.floor(row_bot).astype(np.int64) - 1
        none = r1 < r0
        mid = np.floor(0.5 * (row_top + row_bot)).astype(np.int64)
        ra = np.where(none, mid, r0); rb = np.where(none, mid, r1)
        ok = (ra >= 0) & (rb < self.rows)
        ra = np.clip(ra, 0, self.rows - 1); rb = np.clip(rb, 0, self.rows - 1)
        out = ok.copy()
        for rr in (ra, (ra + rb) // 2, rb):
            got = np.isfinite(self.img[rr, col])
            if allow_no_return is not None:
                got = got | allow_no_return
            out &= got & (self.dil[rr, col] > r + margin)
        return out


def to_sensor(x, y, z, pose):
    c, s = np.cos(pose[3]), np.sin(pose[3])
    dx, dy = x - pose[0], y - pose[1]
    return dx * c + dy * s, -dx * s + dy * c, z - pose[2]


def clear_ghosts(grid, ri: RangeImage, pose, frame: int, cfg: Config):
    """Remove occupied cells the laser saw through.  Returns counters."""
    n_seen = n_cleared = n_stale = 0
    stamp_now = np.uint16(frame & 0xFFFF)
    ground_z = pose[2] - cfg.sensor_height
    for L in grid.levels:
        t, slots, P = grid.level_view(L)
        if len(t) == 0:
            continue
        lo = P["lo"][slots]
        st = P["stamp"][slots]
        ti, ci = np.nonzero((lo > 0) & (st != stamp_now))
        if len(ti) == 0:
            continue
        flat = slots[ti] * P["lo"].shape[1] + ci
        x, y = grid.cell_xy(L, t[ti], ci)
        d = np.hypot(x - pose[0], y - pose[1])
        zt = P["zt"].reshape(-1)[flat]; zg = P["zg"].reshape(-1)[flat]
        zg = np.where(np.isfinite(zg), zg, ground_z)
        zt = np.where(np.isfinite(zt), zt, zg + 0.5)
        res = cfg.level_res(L)
        margin = 0.35 + 0.02 * d + res
        sx, sy, s_hi = to_sensor(x, y, zt - 0.03, pose)
        _, _, s_lo = to_sensor(x, y, np.minimum(zg + 0.12, zt - 0.06), pose)
        hist = P["hist"].reshape(-1, C.NUM_CLASSES)[flat]
        dyn_cls = (hist[:, C.VEHICLE].astype(np.int32) + hist[:, C.PEDESTRIAN]) * 2 > hist[:, 2:].sum(1)
        dynf = (P["dyn"].reshape(-1)[flat] > 0) | dyn_cls
        age0 = (stamp_now - st[ti, ci]).astype(np.uint16)
        relax = dynf & (age0 >= 2)
        seen = (d < cfg.clear_max_range) & ri.seen_through(sx, sy, s_lo, s_hi, margin, relax)
        miss = np.where(dynf, cfg.lo_miss_dynamic, cfg.lo_miss_static)
        lo_f = P["lo"].reshape(-1)
        new_lo = lo_f[flat].astype(np.int16) - np.where(seen, miss, 0)
        age = (stamp_now - P["stamp"].reshape(-1)[flat]).astype(np.uint16)
        stale = (P["dyn"].reshape(-1)[flat] > 0) & (age > cfg.dyn_stale_frames) & ~seen
        new_lo[stale] = 0
        gone = new_lo <= 0
        lo_f[flat] = np.maximum(new_lo, 0).astype(np.int8)
        g = flat[gone]
        P["zt"].reshape(-1)[g] = np.nan
        P["dyn"].reshape(-1)[g] = 0
        hh = P["hist"].reshape(-1, C.NUM_CLASSES)
        hh[g, 2:] = 0
        n_seen += int(seen.sum()); n_cleared += int((gone & seen).sum()); n_stale += int(stale.sum())
    return {"seen_through": n_seen, "cleared": n_cleared, "stale": n_stale}


def clear_trails(grid, tracks, frame: int, cfg: Config):
    """Remove what a moving object left behind along the path it has already travelled.

    The stretch of road directly behind an approaching vehicle is hidden by the vehicle itself,
    so no ray can prove it empty.  The tracker can: the object was there, and it has moved on.
    Cells of a moving class that lie on a moving track's past path, are no longer part of the
    object, and have not been re-observed for a few frames are released.
    """
    paths = []
    for t in tracks:
        if not (t["moving"] and len(t["path"]) >= 2):
            continue
        p = np.asarray(t["path"], dtype=np.float64)
        sp = max(t["speed"], 1e-3)
        back = p[0] - np.array([t["vx"], t["vy"]]) / sp * np.arange(1.0, 41.0)[:, None] * min(sp * 0.1, 1.0)
        half = (0.8 if t["cls"] == C.PEDESTRIAN else 0.5 * max(min(t["ex"], t["ey"]), 1.6) + 0.5)
        keep_out = 0.5 * max(t["ex"], t["ey"]) + 1.0
        paths.append((np.concatenate([back[::-1], p]), half, np.array([t["x"], t["y"]]), keep_out))
    if not paths:
        return 0
    n = 0
    stamp_now = np.uint16(frame & 0xFFFF)
    for L in grid.levels:
        t_, slots, P = grid.level_view(L)
        if len(t_) == 0:
            continue
        lo = P["lo"][slots]
        age = (stamp_now - P["stamp"][slots]).astype(np.uint16)
        ti, ci = np.nonzero((lo > 0) & (age >= 3))
        if len(ti) == 0:
            continue
        flat = slots[ti] * lo.shape[1] + ci
        # any class: a vehicle has just driven through this space, so nothing solid can be standing in it.
        # (far, sparse returns from an approaching truck are often mislabelled; they must go too)
        x, y = grid.cell_xy(L, t_[ti], ci)
        gone = np.zeros(len(x), bool)
        for path, half, now, keep_out in paths:
            reach = half + cfg.level_res(L)
            near = np.flatnonzero((x > path[:, 0].min() - reach) & (x < path[:, 0].max() + reach) &
                                  (y > path[:, 1].min() - reach) & (y < path[:, 1].max() + reach))
            if len(near) == 0:
                continue
            xn, yn = x[near], y[near]
            d2 = ((xn[:, None] - path[None, :, 0]) ** 2 + (yn[:, None] - path[None, :, 1]) ** 2).min(1)
            away = (xn - now[0]) ** 2 + (yn - now[1]) ** 2 > keep_out ** 2
            gone[near[(d2 <= reach ** 2) & away]] = True
        g = flat[gone]
        P["lo"].reshape(-1)[g] = 0
        P["zt"].reshape(-1)[g] = np.nan
        P["dyn"].reshape(-1)[g] = 0
        P["hist"].reshape(-1, C.NUM_CLASSES)[g, 2:] = 0
        n += int(gone.sum())
    return n
