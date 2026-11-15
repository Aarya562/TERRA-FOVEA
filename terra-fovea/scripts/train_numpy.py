"""Train PillarPointNet on the simulator, with NumPy only (no GPU, no framework).

    python scripts/train_numpy.py                 # ~10 minutes on a laptop CPU
    python scripts/train_numpy.py --steps 4000    # longer run

Scenes used for training are randomly generated (random_scene seeds 100+).
The benchmark street (urban_demo) is never seen during training.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from terrafovea import sim, ground, model, classes as C
from terrafovea.config import Config


def make_frames(seeds, per_scene, cfg, rng):
    frames = []
    spec = sim.LidarSpec()
    for sd in seeds:
        sc = sim.random_scene(sd)
        for t in rng.uniform(0, 22, per_scene):
            f = sim.cast(sc, float(t), spec, rng, extra_dropout=float(rng.uniform(0, 0.25)),
                         extra_noise=float(rng.uniform(0, 0.02)))
            w = f["world"]
            g = ground.segment(w[:, 0], w[:, 1], w[:, 2], f["pose"][:2], cfg, sensor_z=f["pose"][2])
            frames.append({"x": w[:, 0].astype(np.float64) - f["pose"][0], "y": w[:, 1].astype(np.float64) - f["pose"][1],
                           "z": w[:, 2], "h": g["h"], "i": f["intensity"], "r": g["range"], "label": f["label"].astype(np.int64),
                           "sel": (g["kind"] != 0) | g["uncertain"]})
    return frames


def batch(fr, rng, selective, augment=True):
    m = fr["sel"] if selective else np.ones(len(fr["x"]), bool)
    x, y = fr["x"][m], fr["y"][m]
    inten = fr["i"][m]
    if augment:
        a = rng.uniform(0, 2 * np.pi); c, s = np.cos(a), np.sin(a)
        x, y = c * x - s * y + rng.uniform(-3, 3), s * x + c * y + rng.uniform(-3, 3)
        inten = np.clip(inten * rng.uniform(0.8, 1.25), 0, 2).astype(np.float32)
    z, h, r, lab = fr["z"][m], fr["h"][m], fr["r"][m], fr["label"][m]
    rep, _ = model.voxel_representatives(x, y, z)
    if augment and len(rep) > 26000:
        rep = rng.choice(rep, 26000, replace=False)
    f0, stats, groups = model.build_features(x[rep], y[rep], z[rep], h[rep], inten[rep], r[rep])
    return f0, stats, groups, lab[rep], r[rep]


def evaluate(p, frames, rng):
    conf = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), np.int64)
    for fr in frames:
        f0, stats, groups, lab, _ = batch(fr, rng, True, augment=False)
        pred = model.forward(p, f0, stats, groups).argmax(1)
        np.add.at(conf, (lab, pred), 1)
    iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
    present = conf.sum(1) > 0
    return float(np.diag(conf).sum() / conf.sum()), float(iou[present].mean()), iou, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--per-scene", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "weights", "pillarpointnet_sim.npz"))
    a = ap.parse_args()
    cfg = Config(); rng = np.random.default_rng(7)
    t0 = time.time()
    train = make_frames(range(100, 100 + a.scenes), a.per_scene, cfg, rng)
    val = make_frames(range(900, 905), 4, cfg, rng)
    print(f"data: {len(train)} train sweeps, {len(val)} val sweeps, {time.time() - t0:.0f}s", flush=True)
    counts = np.zeros(C.NUM_CLASSES)
    for fr in train:
        counts += np.bincount(fr["label"][fr["sel"]], minlength=C.NUM_CLASSES)
    cw = (1.0 / np.sqrt(counts / counts.sum() + 1e-4)); cw = (cw / cw.mean()).astype(np.float32)
    print("class weights", np.round(cw, 2), flush=True)
    p = model.init_params(0); opt = model.Adam(p, lr=3e-3)
    best = (-1, None); ema = None
    for step in range(1, a.steps + 1):
        fr = train[rng.integers(len(train))]
        f0, stats, groups, lab, _ = batch(fr, rng, selective=rng.random() < 0.7)
        loss, g, _ = model.loss_and_grads(p, f0, stats, groups, lab, cw)
        lr = 3e-3 * 0.5 * (1 + np.cos(np.pi * step / a.steps)) + 1e-5
        opt.step(p, g, lr)
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if step % 200 == 0 or step == a.steps:
            acc, miou, iou, _ = evaluate(p, val, rng)
            print(f"step {step:5d} loss {ema:.4f} | val acc {acc:.4f} mIoU {miou:.4f} | {time.time() - t0:.0f}s", flush=True)
            if miou > best[0]:
                best = (miou, {k: v.copy() for k, v in p.items()})
    acc, miou, iou, conf = evaluate(best[1], val, rng)
    print("best val mIoU", round(miou, 4), "per-class IoU", dict(zip(C.NAMES, np.round(iou, 3))))
    model.PillarPointNet(best[1]).save(a.out, val_miou=miou, val_acc=acc, trained_on="simulator random_scene seeds",
                                       class_names=np.array(C.NAMES))
    print("saved", a.out)


if __name__ == "__main__":
    main()
