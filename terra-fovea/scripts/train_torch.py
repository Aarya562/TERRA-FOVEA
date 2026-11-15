"""Train PillarPointNet on SemanticKITTI (or RELLIS-3D in the same layout) with PyTorch on a GPU.

    python scripts/train_torch.py --data /kaggle/input/semantickitti/dataset/sequences --epochs 6

NOT RUN IN THE BUILD ENVIRONMENT.  The machine this project was built on had no PyTorch and no
dataset access, so this script has never been executed.  The architecture, features and weight
file are identical to terrafovea/model.py (which is tested), so the output drops straight into the
pipeline:   python scripts/run_dashboard.py --kitti <sequence> --weights weights/pillarpointnet_kitti.npz
Expect to fix small things on first run; docs/REAL_DATA.md lists what to check.
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from terrafovea import ground, model as M, classes as C
from terrafovea.config import Config
from terrafovea.io_kitti import KittiSequence

TRAIN_SEQ = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQ = ["08"]


class PillarPointNetT(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(M.D_IN, M.H1); self.l2 = nn.Linear(M.H1, M.H2)
        self.l3 = nn.Linear(M.D_CAT, M.H3); self.l4 = nn.Linear(M.H3, M.H4); self.l5 = nn.Linear(M.H4, C.NUM_CLASSES)

    def forward(self, f0, stats, segs):
        a = Fn.relu(self.l2(Fn.relu(self.l1(f0))))
        parts = [a]
        for seg in segs:                               # seg: (N,) pillar index of every point at one scale
            n = int(seg.max()) + 1
            pooled = torch.zeros(n, a.shape[1], device=a.device, dtype=a.dtype)
            pooled = pooled.scatter_reduce(0, seg[:, None].expand(-1, a.shape[1]), a, reduce="amax", include_self=False)
            parts.append(pooled[seg])
        parts.append(stats)
        x = Fn.relu(self.l3(torch.cat(parts, 1)))
        return self.l5(Fn.relu(self.l4(x)))

    def load_npz(self, path):
        z = np.load(path)
        for i, l in enumerate([self.l1, self.l2, self.l3, self.l4, self.l5], 1):
            l.weight.data = torch.from_numpy(z[f"W{i}"].T.copy()); l.bias.data = torch.from_numpy(z[f"b{i}"].copy())

    def save_npz(self, path, **meta):
        p = {}
        for i, l in enumerate([self.l1, self.l2, self.l3, self.l4, self.l5], 1):
            p[f"W{i}"] = l.weight.detach().cpu().numpy().T.copy(); p[f"b{i}"] = l.bias.detach().cpu().numpy().copy()
        np.savez(path, **p, **meta)


def prepare(sweep, cfg, rng, augment):
    """Same preprocessing as the live pipeline: triage, selective mask, voxel representatives, features."""
    T = sweep["T"]; pts = sweep["points"]
    pw = pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
    g = ground.segment(pw[:, 0], pw[:, 1], pw[:, 2].astype(np.float32), sweep["pose"][:2], cfg, sensor_z=sweep["pose"][2])
    m = (g["range"] <= cfg.max_range) & (sweep["label"] >= 0)
    if rng.random() < 0.7:
        m &= (g["kind"] != 0) | g["uncertain"]
    x, y = pw[m, 0] - sweep["pose"][0], pw[m, 1] - sweep["pose"][1]
    if augment:
        a = rng.uniform(0, 2 * np.pi); c, s = np.cos(a), np.sin(a)
        x, y = c * x - s * y, s * x + c * y
    z = pw[m, 2].astype(np.float32); h = g["h"][m]; r = g["range"][m]; inten = sweep["intensity"][m]
    lab = sweep["label"][m].astype(np.int64)
    rep, _ = M.voxel_representatives(x, y, z)
    f0, stats, groups = M.build_features(x[rep], y[rep], z[rep], h[rep], inten[rep], r[rep])
    return f0, stats, [gr[2] for gr in groups], lab[rep], r[rep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding the sequence folders 00, 01, ...")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4, help="use every N-th sweep (neighbouring sweeps are near-duplicates)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init", default=None, help="start from an existing .npz (e.g. the simulator weights)")
    ap.add_argument("--sensor-height", type=float, default=1.73)
    ap.add_argument("--out", default="weights/pillarpointnet_kitti.npz")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(sensor_height=a.sensor_height)
    rng = np.random.default_rng(0)
    seqs = lambda names: [KittiSequence(os.path.join(a.data, s), sensor_height=a.sensor_height) for s in names
                          if os.path.isdir(os.path.join(a.data, s))]
    train, val = seqs(TRAIN_SEQ), seqs(VAL_SEQ)
    items = [(s, i) for s in train for i in range(0, len(s), a.stride)]
    vitems = [(s, i) for s in val for i in range(0, len(s), 20)]
    print(f"{len(items)} training sweeps, {len(vitems)} validation sweeps, device {dev}")
    net = PillarPointNetT().to(dev)
    if a.init:
        net.load_npz(a.init); net.to(dev)
    # class weights from a sample of sweeps
    cnt = np.ones(C.NUM_CLASSES)
    for s, i in [items[j] for j in rng.choice(len(items), min(60, len(items)), replace=False)]:
        l = s[i]["label"]; cnt += np.bincount(l[l >= 0], minlength=C.NUM_CLASSES)
    w = 1 / np.sqrt(cnt / cnt.sum() + 1e-4); w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(items))
    tens = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=dev)
    best = -1
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); run = 0.0
        for k, j in enumerate(rng.permutation(len(items))):
            s, i = items[j]
            f0, stats, segs, lab, _ = prepare(s[i], cfg, rng, True)
            logits = net(tens(f0), tens(stats), [tens(g, torch.long) for g in segs])
            loss = Fn.cross_entropy(logits, tens(lab, torch.long), weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run = 0.98 * run + 0.02 * loss.item() if k else loss.item()
            if k % 200 == 0:
                print(f"epoch {ep} step {k}/{len(items)} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        net.eval(); conf = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), np.int64)
        with torch.no_grad():
            for s, i in vitems:
                f0, stats, segs, lab, _ = prepare(s[i], cfg, np.random.default_rng(1), False)
                pred = net(tens(f0), tens(stats), [tens(g, torch.long) for g in segs]).argmax(1).cpu().numpy()
                np.add.at(conf, (lab, pred), 1)
        iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
        miou = float(iou[conf.sum(1) > 0].mean())
        print(f"epoch {ep}: val mIoU {miou:.4f}", dict(zip(C.NAMES, iou.round(3))))
        if miou > best:
            best = miou
            net.save_npz(a.out, val_miou=miou, trained_on="SemanticKITTI", class_names=np.array(C.NAMES))
            print("saved", a.out)


if __name__ == "__main__":
    main()
