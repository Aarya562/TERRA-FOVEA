"""Synthetic LiDAR with exact ground truth.

A 64-beam spinning LiDAR is ray-cast against an analytic scene (height-field ground,
oriented boxes, vertical cylinders, spheres and rectangular holes).  Every return
carries its true class, object id and a 'negative obstacle' flag, and the scene can
be queried for the true non-traversable footprint at any time.  That is what lets
the benchmark measure accuracy instead of asserting it.

World frame: x forward along the road, y left, z up.  India drives on the left, so
the ego lane is at y = +1.75 and oncoming traffic is at y = -1.75.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import classes as C

F = np.float32


@dataclass
class Scene:
    boxes: list = field(default_factory=list)      # cx cy cz hx hy hz yaw cls oid vx vy
    cyls: list = field(default_factory=list)       # cx cy r z0 z1 cls oid vx vy
    spheres: list = field(default_factory=list)    # cx cy cz r cls oid
    holes: list = field(default_factory=list)      # x0 x1 y0 y1 depth
    road_half: float = 5.5
    flat_half: float = 11.0        # ground is flat (plus roughness) inside this |y|
    hill_amp: float = 0.9
    rough_amp: float = 0.07
    ego_y: float = 1.75
    ego_speed: float = 8.0
    ego_x0: float = 0.0
    name: str = "scene"
    _oid: int = 0

    # ---- builders -------------------------------------------------------
    def new_id(self) -> int:
        self._oid += 1
        return self._oid

    def box(self, x0, x1, y0, y1, z0, z1, cls, yaw=0.0, vel=(0.0, 0.0), oid=None):
        oid = self.new_id() if oid is None else oid
        self.boxes.append([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
                           (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2, yaw, cls, oid, vel[0], vel[1]])
        return oid

    def obox(self, cx, cy, length, width, z0, z1, cls, yaw=0.0, vel=(0.0, 0.0)):
        oid = self.new_id()
        self.boxes.append([cx, cy, (z0 + z1) / 2, length / 2, width / 2, (z1 - z0) / 2, yaw, cls, oid, vel[0], vel[1]])
        return oid

    def cyl(self, cx, cy, r, z0, z1, cls, vel=(0.0, 0.0), oid=None):
        oid = self.new_id() if oid is None else oid
        self.cyls.append([cx, cy, r, z0, z1, cls, oid, vel[0], vel[1]])
        return oid

    def sphere(self, cx, cy, cz, r, cls):
        self.spheres.append([cx, cy, cz, r, cls, self.new_id()])

    def hole(self, x0, x1, y0, y1, depth):
        self.holes.append([x0, x1, y0, y1, depth])

    def vehicle(self, cx, cy, yaw=0.0, vel=(0.0, 0.0), length=4.4, width=1.8, height=1.5):
        """Body plus a narrower cabin, both sharing one object id."""
        oid = self.obox(cx, cy, length, width, 0.25, 0.25 + height * 0.55, C.VEHICLE, yaw, vel)
        c, s = np.cos(yaw), np.sin(yaw)
        off = -0.08 * length
        self.boxes.append([cx + off * c, cy + off * s, 0.25 + height * 0.775, length * 0.27, width * 0.44,
                           height * 0.225, yaw, C.VEHICLE, oid, vel[0], vel[1]])
        return oid

    def pedestrian(self, cx, cy, vel=(0.0, 0.0), height=1.72):
        oid = self.cyl(cx, cy, 0.24, 0.0, height * 0.82, C.PEDESTRIAN, vel)
        self.cyl(cx, cy, 0.12, height * 0.82, height, C.PEDESTRIAN, vel, oid=oid)
        return oid

    def tree(self, cx, cy, trunk_h=3.0, canopy_r=2.2):
        self.cyl(cx, cy, 0.18, 0.0, trunk_h + 0.4, C.POLE)
        self.sphere(cx, cy, trunk_h + canopy_r - 0.5, canopy_r, C.VEGETATION)

    # ---- ground ---------------------------------------------------------
    def ground(self, x, y):
        """Height field: flat road, rough terrain beside it, hills further out."""
        ay = np.abs(y)
        w_rough = np.clip((ay - self.road_half - 1.5) / 1.0, 0.0, 1.0)
        w_hill = np.clip((ay - self.flat_half) / 28.0, 0.0, 1.0)
        w_hill = w_hill * w_hill * (3 - 2 * w_hill)
        rough = self.rough_amp * (np.sin(7.3 * x + 1.1 * y) + np.sin(5.1 * y - 2.3 * x)
                                  + 0.7 * np.sin(11.0 * x + 13.0 * y)) / 2.7
        hill = self.hill_amp * (0.5 - 0.5 * np.cos(x * (2 * np.pi / 90.0) + 0.6 * np.sign(y))) \
            + 0.3 * self.hill_amp * np.sin(y * 0.12)
        return (w_rough * rough + w_hill * hill).astype(F)

    def ground_class(self, x, y):
        return np.where(np.abs(y) <= self.road_half, C.ROAD, C.TERRAIN).astype(np.int8)

    # ---- time-dependent arrays -------------------------------------------
    def arrays(self, t):
        b = np.array(self.boxes, dtype=np.float64).reshape(-1, 11)
        c = np.array(self.cyls, dtype=np.float64).reshape(-1, 9)
        s = np.array(self.spheres, dtype=np.float64).reshape(-1, 6)
        h = np.array(self.holes, dtype=np.float64).reshape(-1, 5)
        b = b.copy(); c = c.copy()
        b[:, 0] += b[:, 9] * t; b[:, 1] += b[:, 10] * t
        c[:, 0] += c[:, 7] * t; c[:, 1] += c[:, 8] * t
        return b, c, s, h

    def ego_pose(self, t):
        return np.array([self.ego_x0 + self.ego_speed * t, self.ego_y, 0.0, 0.0])  # x y z yaw

    # ---- ground truth queries --------------------------------------------
    def gt_blocked(self, x, y, t, veh_h=2.2, step=0.10):
        """True where the footprint at (x, y) cannot be driven over at time t."""
        b, c, s, h = self.arrays(t)
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        out = np.zeros(x.shape, dtype=bool)
        cls = np.full(x.shape, -1, dtype=np.int8)
        dyn = np.zeros(x.shape, dtype=bool)
        for r in b:
            z0, z1 = r[2] - r[5], r[2] + r[5]
            if z1 < step or z0 > veh_h:
                continue
            cs, sn = np.cos(r[6]), np.sin(r[6])
            dx, dy = x - r[0], y - r[1]
            m = (np.abs(dx * cs + dy * sn) <= r[3]) & (np.abs(-dx * sn + dy * cs) <= r[4])
            out |= m; cls[m] = int(r[7]); dyn |= m & ((r[9] != 0) | (r[10] != 0))
        for r in c:
            if r[4] < step or r[3] > veh_h:
                continue
            m = (x - r[0]) ** 2 + (y - r[1]) ** 2 <= r[2] ** 2
            out |= m; cls[m] = int(r[5]); dyn |= m & ((r[7] != 0) | (r[8] != 0))
        for r in s:
            if r[2] - r[3] > veh_h:
                continue
            rr = r[3] ** 2 - (min(veh_h, r[2]) - r[2]) ** 2
            m = (x - r[0]) ** 2 + (y - r[1]) ** 2 <= rr
            out |= m; cls[m] = int(r[4])
        neg = np.zeros(x.shape, dtype=bool)
        for r in h:
            m = (x >= r[0]) & (x <= r[1]) & (y >= r[2]) & (y <= r[3])
            neg |= m
        out |= neg
        return out, cls, dyn, neg

    def moving_objects(self, t):
        """[(oid, cls, x, y, vx, vy)] for every object with non-zero velocity."""
        res = {}
        b, c, _, _ = self.arrays(t)
        for r in b:
            if r[9] or r[10]:
                res[int(r[8])] = (int(r[8]), int(r[7]), r[0], r[1], r[9], r[10])
        for r in c:
            if r[7] or r[8]:
                res[int(r[6])] = (int(r[6]), int(r[5]), r[0], r[1], r[7], r[8])
        return list(res.values())


@dataclass
class LidarSpec:
    n_beams: int = 64
    n_az: int = 1800
    el_top_deg: float = 8.0
    el_bot_deg: float = -24.8
    max_range: float = 100.0
    height: float = 1.9
    range_sigma: float = 0.02
    dropout: float = 0.03
    hz: float = 10.0

    def directions(self):
        el = np.deg2rad(np.linspace(self.el_top_deg, self.el_bot_deg, self.n_beams))
        az = np.linspace(-np.pi, np.pi, self.n_az, endpoint=False)
        ce, se = np.cos(el), np.sin(el)
        d = np.stack([np.outer(ce, np.cos(az)), np.outer(ce, np.sin(az)), np.outer(se, np.ones_like(az))], -1)
        ring = np.repeat(np.arange(self.n_beams), self.n_az)
        return d.reshape(-1, 3), ring


_INTENSITY = np.array([0.22, 0.42, 0.55, 0.72, 0.48, 0.62, 0.36, 0.5], dtype=F)


def cast(scene: Scene, t: float, spec: LidarSpec, rng: np.random.Generator,
         extra_dropout: float = 0.0, extra_noise: float = 0.0):
    """One LiDAR sweep at time t.  Returns a dict of per-point arrays in the SENSOR frame."""
    pose = scene.ego_pose(t)
    d_ego, ring = spec.directions()
    cy, sy = np.cos(pose[3]), np.sin(pose[3])
    d = d_ego.copy()
    d[:, 0] = cy * d_ego[:, 0] - sy * d_ego[:, 1]
    d[:, 1] = sy * d_ego[:, 0] + cy * d_ego[:, 1]
    o = np.array([pose[0], pose[1], pose[2] + spec.height])
    n = d.shape[0]
    best_t = np.full(n, np.inf)
    best_cls = np.full(n, -1, dtype=np.int8)
    best_oid = np.zeros(n, dtype=np.int32)
    neg = np.zeros(n, dtype=bool)
    b, c, s, h = scene.arrays(t)
    R = spec.max_range

    # ---- ground (fixed-point iteration on the height field) ----
    down = d[:, 2] < -1e-3
    tg = np.full(n, np.inf)
    dz = d[down, 2]
    tt = o[2] / -dz
    for _ in range(5):
        gx = o[0] + tt * d[down, 0]; gy = o[1] + tt * d[down, 1]
        tt = (o[2] - scene.ground(gx, gy)) / -dz
    tg[down] = tt
    with np.errstate(invalid="ignore"):
        gx = o[0] + tg * d[:, 0]; gy = o[1] + tg * d[:, 1]
    gcls = np.full(n, -1, dtype=np.int8)
    ok = np.isfinite(tg)
    gcls[ok] = scene.ground_class(gx[ok], gy[ok])
    # holes: a ray that reaches the surface inside a hole carries on to its floor or far wall
    for r in h:
        m = ok & (gx >= r[0]) & (gx <= r[1]) & (gy >= r[2]) & (gy <= r[3])
        if not m.any():
            continue
        lo = np.array([r[0], r[2], -r[4]]); hi = np.array([r[1], r[3], 0.0])
        dd = d[m]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o) / dd; t2 = (hi - o) / dd
        tex = np.minimum(np.minimum(np.maximum(t1[:, 0], t2[:, 0]), np.maximum(t1[:, 1], t2[:, 1])),
                         np.maximum(t1[:, 2], t2[:, 2]))
        tg[m] = tex; neg[m] = True
    best_t = tg.copy(); best_cls = gcls.copy()

    # ---- oriented boxes ----
    if len(b):
        near = np.hypot(b[:, 0] - o[0], b[:, 1] - o[1]) - np.hypot(b[:, 3], b[:, 4]) < R
        for r in b[near]:
            cs, sn = np.cos(r[6]), np.sin(r[6])
            ox, oy, oz = o[0] - r[0], o[1] - r[1], o[2] - r[2]
            ob = np.array([ox * cs + oy * sn, -ox * sn + oy * cs, oz])
            db = np.stack([d[:, 0] * cs + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * cs, d[:, 2]], 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / db
                t1 = (-r[3:6] - ob) * inv; t2 = (r[3:6] - ob) * inv
            tn = np.minimum(t1, t2).max(1); tf = np.maximum(t1, t2).min(1)
            m = (tf >= tn) & (tn > 0) & (tn < best_t)
            best_t[m] = tn[m]; best_cls[m] = int(r[7]); best_oid[m] = int(r[8]); neg[m] = False

    # ---- vertical cylinders (side + top cap) ----
    if len(c):
        near = np.hypot(c[:, 0] - o[0], c[:, 1] - o[1]) < R + 1
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        for r in c[near]:
            ox, oy = o[0] - r[0], o[1] - r[1]
            bb = 2 * (ox * d[:, 0] + oy * d[:, 1]); cc = ox * ox + oy * oy - r[2] ** 2
            disc = bb * bb - 4 * a * cc
            m = disc > 0
            ts = np.full(n, np.inf)
            ts[m] = (-bb[m] - np.sqrt(disc[m])) / (2 * a[m])
            z = o[2] + ts * d[:, 2]
            m = (ts > 0) & (z >= r[3]) & (z <= r[4]) & (ts < best_t)
            best_t[m] = ts[m]; best_cls[m] = int(r[5]); best_oid[m] = int(r[6]); neg[m] = False
            with np.errstate(divide="ignore", invalid="ignore"):
                tc = (r[4] - o[2]) / d[:, 2]
            px = ox + tc * d[:, 0]; py = oy + tc * d[:, 1]
            m = (d[:, 2] < 0) & (tc > 0) & (px * px + py * py <= r[2] ** 2) & (tc < best_t)
            best_t[m] = tc[m]; best_cls[m] = int(r[5]); best_oid[m] = int(r[6]); neg[m] = False

    # ---- spheres ----
    for r in s:
        if np.hypot(r[0] - o[0], r[1] - o[1]) > R + r[3]:
            continue
        oc = o - r[:3]
        bb = 2 * (d @ oc); cc = oc @ oc - r[3] ** 2
        disc = bb * bb - 4 * cc
        m = disc > 0
        ts = np.full(n, np.inf)
        ts[m] = (-bb[m] - np.sqrt(disc[m])) / 2
        m = (ts > 0) & (ts < best_t)
        best_t[m] = ts[m]; best_cls[m] = int(r[4]); best_oid[m] = int(r[5]); neg[m] = False

    # ---- sensor effects ----
    sig = spec.range_sigma + extra_noise
    tn = best_t + rng.normal(0.0, sig, n)
    # foliage is porous: canopy returns scatter in depth
    veg = best_cls == C.VEGETATION
    tn[veg] += rng.uniform(0, 0.5, int(veg.sum()))
    keep = np.isfinite(best_t) & (best_t < R) & (best_t > 1.2) & (rng.random(n) > spec.dropout + extra_dropout)
    pts_w = o + tn[keep, None] * d[keep]
    pts = tn[keep, None] * d_ego[keep]
    cls = best_cls[keep]
    inten = _INTENSITY[cls] * (0.75 + 0.5 * rng.random(cls.shape[0], dtype=F)) / (1 + 0.004 * tn[keep].astype(F))
    return {
        "points": pts.astype(F),                # sensor frame (x fwd, y left, z up, origin at sensor)
        "intensity": inten.astype(F),
        "ring": ring[keep].astype(np.int16),
        "label": cls.astype(np.int8),
        "oid": best_oid[keep],
        "neg": neg[keep],
        "pose": np.array([pose[0], pose[1], pose[2] + spec.height, pose[3]], dtype=np.float64),
        "t": float(t),
        "world": pts_w.astype(F),
    }


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def urban_demo() -> Scene:
    """The hand-designed test drive: every hazard the problem statement names, in one street."""
    sc = Scene(name="urban_demo")
    rh = sc.road_half
    # raised footpaths (kerb = 15 cm step), with a side-street gap at x = 95..105
    for x0, x1 in [(-60, 95), (105, 320)]:
        sc.box(x0, x1, rh, rh + 2.6, 0, 0.15, C.TERRAIN)
        sc.box(x0, x1, -rh - 2.6, -rh, 0, 0.15, C.TERRAIN)
    # buildings on the left, behind the footpath
    for x0, ln, hgt in [(-40, 22, 9), (-12, 18, 7), (12, 26, 11), (44, 20, 8), (70, 22, 12),
                        (108, 30, 9), (144, 24, 7), (174, 30, 10), (210, 30, 8), (246, 40, 11)]:
        sc.box(x0, x0 + ln, rh + 3.4, rh + 12, 0, hgt, C.BUILDING)
    # street-light poles on the left footpath, trees on the right footpath (canopy overhangs the lane)
    for x in np.arange(-30, 300, 25.0):
        sc.cyl(x, rh + 0.5, 0.09, 0, 7.0, C.POLE)
    for x in np.arange(-22, 300, 21.0):
        if 92 < x < 108:
            continue
        sc.tree(x, -rh - 1.0)
    # parked cars on the ego side kerb
    for x in [18.0, 24.2, 37.0, 52.0, 58.5, 118.0, 124.5, 131.0, 176.0]:
        sc.vehicle(x, rh - 1.05)
    # ---- dynamic actors ----
    sc.vehicle(120.0, -1.75, np.pi, vel=(-9.0, 0.0))                       # oncoming car
    sc.vehicle(230.0, -1.75, np.pi, vel=(-7.0, 0.0), length=7.5, width=2.4, height=3.0)   # oncoming truck
    sc.vehicle(24.0, 1.75, 0.0, vel=(7.0, 0.0))                            # slower lead vehicle
    sc.vehicle(100.0, -44.0, np.pi / 2, vel=(0.0, 6.5))                    # crosses at the side street
    sc.pedestrian(10.0, rh + 1.6, vel=(1.3, 0.0))                          # walking along the footpath
    sc.pedestrian(47.2, rh + 0.9, vel=(0.0, -1.15))                        # steps out between parked cars
    sc.pedestrian(142.0, -rh - 1.2, vel=(0.0, 1.3))                        # crosses from the right, later
    sc.pedestrian(64.0, -rh - 1.4)                                         # standing still
    sc.pedestrian(160.0, rh + 1.5, vel=(-1.2, 0.0))
    # ---- negative obstacles ----
    sc.hole(30.0, 30.9, 1.2, 2.1, 0.18)        # pothole in the ego lane
    sc.hole(70.0, 71.5, -0.6, 0.9, 0.35)       # crater on the centre line
    sc.hole(121.0, 121.5, 1.5, 2.0, 0.12)      # small, shallow pothole (hard case)
    sc.hole(48.0, 58.0, -9.6, -9.0, 0.6)       # trench in the rough ground on the right
    # ---- overhead structures ----
    sc.cyl(60.0, rh + 0.9, 0.16, 0, 5.6, C.POLE); sc.cyl(60.0, -rh - 0.9, 0.16, 0, 5.6, C.POLE)
    sc.box(59.8, 60.2, -rh - 0.9, rh + 0.9, 4.9, 5.6, C.OTHER)             # sign gantry: passable
    sc.cyl(150.0, -rh - 0.5, 0.12, 0, 1.9, C.POLE)
    sc.box(149.92, 150.08, -rh - 0.5, -0.2, 1.62, 1.8, C.OTHER)            # low boom over the other lane: blocks
    # ---- static clutter ----
    sc.box(110, 136, -rh - 4.2, -rh - 4.0, 0, 2.2, C.BUILDING)             # boundary wall
    for x in [80.0, 81.6, 83.2, 84.8]:
        sc.box(x, x + 1.2, -rh + 0.15, -rh + 0.65, 0, 0.8, C.OTHER)        # jersey barriers
    sc.box(88.0, 88.3, 2.1, 2.4, 0, 0.25, C.OTHER)                         # small debris in the ego lane
    sc.box(196.0, 196.4, 1.2, 1.6, 0, 0.3, C.OTHER)
    return sc


def random_scene(seed: int) -> Scene:
    """Randomised street for training/validation.  Same ingredients, different layout."""
    rng = np.random.default_rng(seed)
    sc = Scene(name=f"random_{seed}")
    sc.road_half = float(rng.uniform(3.5, 7.0))
    sc.flat_half = sc.road_half + float(rng.uniform(3.0, 9.0))
    sc.hill_amp = float(rng.uniform(0.2, 1.4))
    sc.rough_amp = float(rng.uniform(0.06, 0.12))
    sc.ego_y = float(rng.uniform(-0.3, 0.45) * sc.road_half)
    sc.ego_speed = float(rng.uniform(4.0, 12.0))
    rh = sc.road_half
    kerb = float(rng.uniform(0.10, 0.2))
    for side in (1, -1):
        if rng.random() < 0.75:
            w = rng.uniform(1.5, 3.5)
            y0, y1 = sorted([side * rh, side * (rh + w)])
            sc.box(-80, 320, y0, y1, 0, kerb, C.TERRAIN)
        x = -60.0
        builds = rng.random() < 0.6
        while x < 300:
            ln = rng.uniform(8, 35)
            if builds and rng.random() < 0.8:
                y0, y1 = sorted([side * (rh + rng.uniform(3.2, 6)), side * (rh + rng.uniform(10, 16))])
                sc.box(x, x + ln, y0, y1, 0, rng.uniform(3, 14), C.BUILDING)
            elif rng.random() < 0.35:
                yw = side * (rh + rng.uniform(3.5, 9))
                sc.box(x, x + ln, yw - 0.12, yw + 0.12, 0, rng.uniform(1.2, 3.0), C.BUILDING)
            x += ln + rng.uniform(2, 14)
        for x in np.arange(-50, 300, rng.uniform(14, 32)):
            if rng.random() < 0.5:
                sc.cyl(x + rng.uniform(-2, 2), side * (rh + rng.uniform(0.3, 1.2)), rng.uniform(0.06, 0.14), 0,
                       rng.uniform(3, 8), C.POLE)
            else:
                sc.tree(x + rng.uniform(-3, 3), side * (rh + rng.uniform(0.6, 5.0)), rng.uniform(2.2, 4.0),
                        rng.uniform(1.4, 2.8))
    for _ in range(int(rng.integers(6, 16))):          # parked + moving vehicles
        x = rng.uniform(-40, 280); lane = rng.choice([-1, 1])
        big = rng.random() < 0.2
        L, W, H = (rng.uniform(6, 9), rng.uniform(2.2, 2.6), rng.uniform(2.6, 3.4)) if big else \
                  (rng.uniform(3.6, 4.9), rng.uniform(1.6, 1.95), rng.uniform(1.35, 1.8))
        if rng.random() < 0.5:
            sc.vehicle(x, lane * (rh - W / 2 - 0.15), rng.choice([0, np.pi]), length=L, width=W, height=H)
        else:
            v = lane * rng.uniform(3, 12)
            sc.vehicle(x, -lane * rh * 0.4 + sc.ego_y * 0 + rng.uniform(-0.3, 0.3), 0 if v > 0 else np.pi,
                       vel=(v, 0.0), length=L, width=W, height=H)
    for _ in range(int(rng.integers(1, 4))):            # crossing vehicles
        x = rng.uniform(40, 250); sgn = rng.choice([-1, 1])
        sc.vehicle(x, -sgn * rng.uniform(20, 60), sgn * np.pi / 2, vel=(0.0, sgn * rng.uniform(3, 8)))
    for _ in range(int(rng.integers(8, 20))):           # pedestrians
        x = rng.uniform(-30, 280); y = rng.uniform(-rh - 3, rh + 3)
        mode = rng.random()
        vel = (0.0, 0.0) if mode < 0.3 else ((rng.choice([-1, 1]) * rng.uniform(0.8, 1.8), 0.0) if mode < 0.65
                                             else (0.0, rng.choice([-1, 1]) * rng.uniform(0.8, 1.8)))
        sc.pedestrian(x, y, vel=vel, height=rng.uniform(1.5, 1.9))
    for _ in range(int(rng.integers(3, 8))):            # holes
        x = rng.uniform(5, 280); y = rng.uniform(-rh + 0.5, rh - 1.5)
        sc.hole(x, x + rng.uniform(0.4, 1.8), y, y + rng.uniform(0.4, 1.6), rng.uniform(0.1, 0.5))
    for _ in range(int(rng.integers(4, 12))):           # clutter
        x = rng.uniform(0, 280); y = rng.uniform(-rh, rh)
        sc.box(x, x + rng.uniform(0.2, 1.4), y, y + rng.uniform(0.2, 0.7), 0, rng.uniform(0.2, 1.0), C.OTHER)
    for _ in range(int(rng.integers(0, 3))):            # gantries / booms
        x = rng.uniform(30, 260); z0 = rng.choice([rng.uniform(1.4, 2.0), rng.uniform(4.2, 5.5)])
        sc.box(x - 0.15, x + 0.15, -rh - 0.8, rh + 0.8, z0, z0 + rng.uniform(0.15, 0.7), C.OTHER)
        sc.cyl(x, rh + 0.8, 0.15, 0, z0 + 0.5, C.POLE); sc.cyl(x, -rh - 0.8, 0.15, 0, z0 + 0.5, C.POLE)
    return sc


def generate(scene: Scene, n_frames: int, spec: LidarSpec | None = None, seed: int = 0, start: float = 0.0,
             extra_dropout: float = 0.0, extra_noise: float = 0.0):
    """Yield n_frames consecutive sweeps."""
    spec = spec or LidarSpec()
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        yield cast(scene, start + i / spec.hz, spec, rng, extra_dropout, extra_noise)
