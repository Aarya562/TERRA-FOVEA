"""End-to-end behaviour on a short simulated drive (uses the shipped network weights)."""
import os, tempfile
import numpy as np
from terrafovea import sim, classes as C
from terrafovea.config import Config
from terrafovea.model import PillarPointNet
from terrafovea.pipeline import Pipeline, cell_state
from terrafovea.io_kitti import KittiSequence, write_sequence

ROOT = os.path.join(os.path.dirname(__file__), "..")
_cache = {}


def _run(n=45, **over):
    key = (n, tuple(sorted(over.items())))
    if key not in _cache:
        net = PillarPointNet.load(os.path.join(ROOT, "weights", "pillarpointnet_sim.npz"))
        scene = sim.urban_demo()
        pipe = Pipeline(Config(**over), net=net, fov=(8.0, -24.8))
        outs = []; sweeps = []
        for sw in sim.generate(scene, n, seed=1):
            outs.append(pipe.process(sw)); sweeps.append(sw)
        _cache[key] = (scene, pipe, outs, sweeps)
    return _cache[key]


def test_pothole_is_found_and_nothing_else_is_called_a_pit():
    scene, pipe, outs, _ = _run()
    c = pipe.grid.cells(); st = cell_state(c, pipe.cfg, 0.0)
    hx, hy = c["x"][st["hole"]], c["y"][st["hole"]]
    assert len(hx) > 50                                            # the pothole at x = 30 m is in the map
    inside = np.zeros(len(hx), bool)
    for x0, x1, y0, y1, _ in scene.holes:
        inside |= (hx > x0 - 0.15) & (hx < x1 + 0.15) & (hy > y0 - 0.15) & (hy < y1 + 0.15)
    assert inside.mean() > 0.97                                    # and there are no phantom pits


def test_moving_objects_are_tracked_and_parked_cars_are_not():
    scene, pipe, outs, sweeps = _run()
    o = outs[-1]
    mov = [t for t in o["tracks"] if t["moving"]]
    lead = [t for t in mov if t["cls"] == C.VEHICLE and abs(t["speed"] - 7.0) < 1.0 and abs(t["y"] - 1.75) < 1.0]
    assert lead, "the lead vehicle (7 m/s) must be tracked as moving"
    parked = [t for t in o["tracks"] if t["confirmed"] and t["cls"] == C.VEHICLE and abs(t["y"] - 4.45) < 0.8]
    assert parked and not any(t["moving"] for t in parked)


def test_adaptive_map_is_much_smaller_than_uniform_fine():
    _, _, outs, _ = _run()
    _, _, fine, _ = _run(n=12, policy="uniform_fine")
    assert fine[-1]["memory_bytes"] > 8 * outs[11]["memory_bytes"]


def test_risk_policy_refines_the_path_beyond_the_near_band():
    _, pipe, outs, _ = _run()
    o = outs[-1]; g = pipe.grid; ts = pipe.cfg.tile_size
    ids = g.tile_ids()
    cx = (g.tx[ids] + 0.5) * ts - o["pose"][0]; cy = (g.ty[ids] + 0.5) * ts - o["pose"][1]
    path = (cx > 12) & (cx < 28) & (np.abs(cy) < 1.5)              # in the lane, beyond the 10 m fine band
    side = (cx > 12) & (cx < 28) & (cy < -12) & (cy > -20)         # same range, off to the side
    assert g.tlevel[ids][path].mean() < g.tlevel[ids][side].mean() - 0.5


def test_ghost_clearing_removes_cells():
    _, _, outs, _ = _run()
    assert sum(o["clearing"]["cleared"] + o["clearing"]["trail"] for o in outs) > 200


def test_kitti_layout_round_trip():
    _, _, _, sweeps = _run()
    with tempfile.TemporaryDirectory() as d:
        write_sequence(d, sweeps[:3])
        seq = KittiSequence(d, sensor_height=1.9)
        assert len(seq) == 3
        s = seq[2]
        assert np.allclose(s["points"], sweeps[2]["points"])
        assert np.array_equal(s["label"], sweeps[2]["label"])
        assert np.allclose(s["pose"], sweeps[2]["pose"], atol=1e-6)
        out = Pipeline(Config()).process(s)                         # the pipeline accepts it as-is
        assert out["n_points"] == len(s["points"])


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
