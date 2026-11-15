"""Reproduce every number in the report.

    python scripts/run_benchmark.py            # full run (~10-15 min on a laptop)
    python scripts/run_benchmark.py --frames 60 --quick

All variants see exactly the same sweeps of the same drive (urban_demo), a street
the network never saw in training.  Results go to results/benchmark.json.
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from terrafovea import sim, metrics, classes as C
from terrafovea.config import Config
from terrafovea.grid import BYTES_PER_CELL
from terrafovea.model import PillarPointNet
from terrafovea.pipeline import Pipeline, STAGES

VARIANTS = {
    "B1_uniform_5cm":   dict(policy="uniform_fine"),
    "B2_uniform_40cm":  dict(policy="uniform_coarse"),
    "B3_distance_only": dict(policy="distance"),
    "B4_terra_fovea":   dict(policy="risk"),
    "A1_no_ray_clearing": dict(policy="risk", ray_clearing=False),
    "A2_ai_on_all_points": dict(policy="risk", selective_ai=False),
    "A3_geometry_only": dict(policy="risk", use_network=False),
}


def dynamic_score(scene, t, tracks, pose, sweep):
    """Match tracks flagged 'moving' against the objects that truly move."""
    truth = []
    for oid, cls, x, y, vx, vy in scene.moving_objects(t):
        d = np.hypot(x - pose[0], y - pose[1])
        if d < 50 and int((sweep["oid"] == oid).sum()) >= 12:
            truth.append((x, y, np.hypot(vx, vy)))
    every = [(x, y) for _, _, x, y, _, _ in scene.moving_objects(t)]
    mov = [tr for tr in tracks if tr["moving"] and tr["visible"]]
    false_alarms = sum(1 for tr in mov if not any(np.hypot(tr["x"] - x, tr["y"] - y) < 3.0 for x, y in every))
    tp = 0; verr = []
    used = set()
    for x, y, sp in truth:
        best = None
        for i, tr in enumerate(mov):
            d = np.hypot(tr["x"] - x, tr["y"] - y)
            if i not in used and d < 3.0 and (best is None or d < best[1]):
                best = (i, d)
        if best:
            used.add(best[0]); tp += 1; verr.append(abs(mov[best[0]]["speed"] - sp))
    return tp, len(truth), tp + false_alarms, verr


def run_variant(name, overrides, sweeps, scene, net, eval_every=5, warmup=15, log=print):
    cfg = Config(**overrides)
    pipe = Pipeline(cfg, net=net, fov=(8.0, -24.8))
    conf = np.zeros((len(metrics.BANDS), C.NUM_CLASSES, C.NUM_CLASSES + 1), np.int64)
    timing = {k: [] for k in STAGES + ("total",)}
    mem, cells, per_level, n_net, npts = [], [], [], [], []
    mm = []
    dyn = [0, 0, 0]; verr = []
    cleared = 0
    t0 = time.time()
    for i, sw in enumerate(sweeps):
        o = pipe.process(sw)
        if i >= 3:
            for k in timing:
                timing[k].append(o["timing"][k] * 1000)
        mem.append(o["memory_bytes"]); cells.append(o["n_cells"]); per_level.append(o["cells_per_level"].tolist())
        n_net.append(o["n_net"]); npts.append(o["n_points"])
        cleared += o["clearing"]["cleared"] + o["clearing"]["stale"] + o["clearing"]["trail"]
        if i >= warmup:
            conf += metrics.seg_confusion(sw["label"], o["cls"], o["range"])
            tp, nt, nm, ve = dynamic_score(scene, sw["t"], o["tracks"], o["pose"], sw)
            dyn[0] += tp; dyn[1] += nt; dyn[2] += nm; verr += ve
            if (i - warmup) % eval_every == 0:
                m, _ = metrics.map_vs_truth(pipe.grid, scene, sw["t"], o["pose"], cfg)
                m["frame"] = i
                mm.append(m)
    wall = time.time() - t0
    res = {"name": name, "config": overrides, "frames": len(sweeps),
           "memory_mb_mean": float(np.mean(mem[warmup:]) / 1e6), "memory_mb_peak": float(np.max(mem) / 1e6),
           "memory_mb_series": [round(v / 1e6, 3) for v in mem],
           "cells_mean": float(np.mean(cells[warmup:])), "cells_per_level_last": per_level[-1],
           "timing_ms": {k: float(np.mean(v)) for k, v in timing.items()},
           "timing_ms_p95": {k: float(np.percentile(v, 95)) for k, v in timing.items()},
           "fps": float(1000.0 / np.mean(timing["total"])),
           "net_points_mean": float(np.mean(n_net)), "points_mean": float(np.mean(npts)),
           "cells_cleared": int(cleared), "wall_s": wall,
           "map": {k: float(np.mean([m[k] for m in mm])) for k in mm[0] if k != "frame"},
           "map_series": mm,
           "dynamic": {"recall": dyn[0] / max(dyn[1], 1), "precision": dyn[0] / max(dyn[2], 1),
                       "truth": dyn[1], "speed_err_ms": float(np.mean(verr)) if verr else None},
           "seg": {}}
    tot = conf.sum(0)
    st = metrics.conf_stats(tot)
    res["seg"]["overall"] = {"acc": st["acc"], "miou": st["miou"], "iou": dict(zip(C.NAMES, st["iou"].round(4).tolist()))}
    g = metrics.group_confusion(tot)
    res["seg"]["group_recall"] = dict(zip(C.GROUP_NAMES, (np.diag(g[:, :3]) / np.maximum(g.sum(1), 1)).round(4).tolist()))
    res["seg"]["bands"] = []
    for b, (lo, hi) in enumerate(metrics.BANDS):
        s = metrics.conf_stats(conf[b]); gb = metrics.group_confusion(conf[b])
        res["seg"]["bands"].append({"band": f"{lo}-{hi} m", "acc": s["acc"], "miou": s["miou"],
                                    "group_acc": float(np.diag(gb[:, :3]).sum() / max(gb.sum(), 1)),
                                    "group_recall": (np.diag(gb[:, :3]) / np.maximum(gb.sum(1), 1)).round(4).tolist(),
                                    "points": int(conf[b].sum())})
    log(f"{name:22s} mem {res['memory_mb_mean']:8.2f} MB | {res['fps']:5.1f} FPS | inflation {res['map']['inflation_m2']:.2f} m2 "
        f"| phantom {res['map']['phantom_m2']:.2f} m2 | edge {res['map']['edge_err_cm']:.1f} cm | acc {st['acc']:.3f} mIoU {st['miou']:.3f} | {wall:.0f}s")
    return res


def sparse_voxel_baseline(sweeps, cfg, res=0.05):
    """Measured size of a sparse 5 cm 3D voxel map of the same drive in the same 100 m window."""
    keys = np.zeros(0, np.int64)
    off = 1 << 20
    series = []
    for i, sw in enumerate(sweeps):
        w = sw["world"]
        k = ((np.floor(w[:, 0] / res).astype(np.int64) + off) << 42) | ((np.floor(w[:, 1] / res).astype(np.int64) + off) << 21) \
            | (np.floor(w[:, 2] / res).astype(np.int64) + off)
        keys = np.union1d(keys, k)
        if i % 5 == 0:
            x = ((keys >> 42) - off) * res; y = (((keys >> 21) & ((1 << 21) - 1)) - off) * res
            keys = keys[np.hypot(x - sw["pose"][0], y - sw["pose"][1]) <= cfg.keep_radius]
        series.append(len(keys))
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--quick", action="store_true", help="skip the uniform 5 cm baseline and the ablations")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "pillarpointnet_sim.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reference", "benchmark.json"))
    a = ap.parse_args()
    net = PillarPointNet.load(a.weights)
    scene = sim.urban_demo()
    t0 = time.time()
    sweeps = list(sim.generate(scene, a.frames, seed=1))
    print(f"simulated {len(sweeps)} sweeps ({np.mean([len(s['points']) for s in sweeps]):.0f} points each) in {time.time() - t0:.0f}s", flush=True)
    out = {"frames": a.frames, "scene": scene.name, "bytes_per_cell": BYTES_PER_CELL, "variants": {},
           "network_params": net.n_params()}
    names = ["B4_terra_fovea", "B3_distance_only", "B2_uniform_40cm"] if a.quick else list(VARIANTS)
    for n in names:
        out["variants"][n] = run_variant(n, VARIANTS[n], sweeps, scene, net, log=lambda s: print(s, flush=True))
    cfg = Config()
    side = int(2 * cfg.max_range / cfg.base_res)
    vox = sparse_voxel_baseline(sweeps, cfg)
    out["baselines"] = {
        "dense_3d_voxel_5cm_mb": side * side * int(5.0 / cfg.base_res) * 1 / 1e6,          # 1 byte per voxel, 5 m tall
        "dense_2p5d_5cm_mb": side * side * BYTES_PER_CELL / 1e6,
        "sparse_3d_voxel_5cm_mb": float(np.mean(vox[15:]) * 12 / 1e6),                      # 8-byte key + 4-byte payload
        "sparse_3d_voxel_count": float(np.mean(vox[15:])),
    }
    # robustness: heavy dropout + extra range noise (dust / rain stand-in)
    if not a.quick:
        rs = list(sim.generate(scene, a.frames, seed=2, extra_dropout=0.30, extra_noise=0.03))
        out["robustness"] = run_variant("R1_dropout30_noise3cm", dict(policy="risk"), rs, scene, net,
                                        log=lambda s: print(s, flush=True))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
