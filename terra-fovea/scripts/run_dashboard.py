"""Live dashboard.  Runs the pipeline and serves the 3D map to a browser, fully offline.

    python scripts/run_dashboard.py                       # open http://localhost:8765
    python scripts/run_dashboard.py --seq /data/sequences/08 --sensor-height 1.73    # SemanticKITTI
    python scripts/run_dashboard.py --engine python       # force the NumPy reference pipeline
    python scripts/run_dashboard.py --record results/replay.html --frames 160   # self-contained replay file

Uses the C++ core when it has been built (build.sh / build.bat), otherwise the NumPy reference.
Only the Python standard library is used for the server, so there is nothing to install at the venue.
"""
import argparse, base64, gzip, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
from terrafovea import sim
from terrafovea.config import Config
from terrafovea.model import PillarPointNet
from terrafovea.pipeline import Pipeline
from terrafovea.export import build_frame


def load_baselines():
    """Reference memory figures measured by run_benchmark.py, shown beside the live number."""
    p = os.path.join(ROOT, "results", "reference", "benchmark.json")
    if not os.path.exists(p):
        return {}
    r = json.load(open(p))
    b = dict(r.get("baselines", {}))
    u = r["variants"].get("B1_uniform_5cm")
    if u:
        b["uniform_5cm_sparse_mb"] = u["memory_mb_mean"]
    return b


def to_T(pose):
    c, s = np.cos(pose[3]), np.sin(pose[3])
    return np.array([[c, -s, 0, pose[0]], [s, c, 0, pose[1]], [0, 0, 1, pose[2]], [0, 0, 0, 1.0]])


class Source:
    """Endless stream of sweeps: a SemanticKITTI-layout folder, or the built-in simulator."""

    def __init__(self, args):
        self.args = args
        self.cache = []
        if args.seq:
            from terrafovea.io_kitti import KittiSequence
            self.seq = KittiSequence(args.seq, sensor_height=args.sensor_height)
            self.n = min(len(self.seq), args.frames) if args.frames else len(self.seq)
            self.fov = (None, None)
            self.name = os.path.basename(os.path.normpath(args.seq))
        else:
            self.scene = sim.urban_demo()
            self.spec = sim.LidarSpec()
            self.rng = np.random.default_rng(1)
            self.n = args.frames or 200
            self.fov = (self.spec.el_top_deg, self.spec.el_bot_deg)
            self.name = "simulated street"

    def get(self, i):
        while len(self.cache) <= i:
            k = len(self.cache)
            if self.args.seq:
                self.cache.append(self.seq[k])
            else:
                f = sim.cast(self.scene, k / self.spec.hz, self.spec, self.rng)
                self.cache.append({k_: f[k_] for k_ in ("points", "intensity", "pose", "t")})
        return self.cache[i]


class Backend:
    """One interface over the two engines: the C++ core and the NumPy reference pipeline."""

    def __init__(self, kind, src, weights, view):
        self.kind, self.src, self.view = kind, src, view
        self.seq_no = 0
        if kind == "cpp":
            from terrafovea import cpp_bridge
            self.cpp = cpp_bridge
            self.extra = {"baselines": cpp_bridge.baselines(), "source": src.name}
        else:
            self.net = PillarPointNet.load(weights) if os.path.exists(weights) else None
            self.extra = {"baselines": load_baselines(), "source": src.name, "engine": "NumPy reference"}
        self.reset({"policy": "risk", "ray_clearing": True})

    def reset(self, ov):
        self.ov = dict(ov)
        if self.kind == "cpp":
            self.eng = self.cpp.make_engine(ov["policy"], ov["ray_clearing"])
        else:
            self.pipe = Pipeline(Config(policy=ov["policy"], ray_clearing=ov["ray_clearing"]), net=self.net, fov=self.src.fov)

    def step(self, sweep, lod=None):
        self.seq_no += 1
        extra = dict(self.extra, seq=self.seq_no)
        if self.kind == "cpp":
            pts = np.concatenate([sweep["points"], sweep["intensity"][:, None]], 1).astype(np.float32)
            self.eng.process(pts, sweep["T"] if "T" in sweep else to_T(sweep["pose"]), sweep["t"])
            return self.cpp.build_frame(self.eng, self.seq_no, sweep["t"], self.ov["policy"], self.ov["ray_clearing"],
                                        view_radius=self.view, extra=extra, ground_lod=lod or 0.2)
        out = self.pipe.process(sweep)
        kw = {"ground_lod": ((12.0, 2), (1e9, 3))} if lod else {}
        return build_frame(self.pipe, out, view_radius=self.view, extra=extra, **kw)


def pick_engine(args):
    if args.engine in ("auto", "cpp"):
        try:
            from terrafovea import cpp_bridge  # noqa: F401
            return "cpp"
        except (SystemExit, OSError) as e:
            if args.engine == "cpp":
                sys.exit("C++ core not available (%s). Build it with build.sh / build.bat." % e)
            print("C++ core not built - using the NumPy reference pipeline (slower). Run build.sh or build.bat for full speed.")
    return "python"


class Runner(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.src = Source(args)
        self.backend = Backend(pick_engine(args), self.src, args.weights, args.view)
        self.overrides = {"policy": "risk", "ray_clearing": True}
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.paused = False
        self.restart = False

    def run(self):
        i = 0
        while True:
            if self.restart:
                with self.lock:
                    ov = dict(self.overrides); self.restart = False
                self.backend.reset(ov); i = 0
            if self.paused:
                time.sleep(0.05); continue
            t0 = time.perf_counter()
            buf = self.backend.step(self.src.get(i))
            with self.lock:
                self.latest = buf; self.seq = self.backend.seq_no
            i += 1
            if i >= self.src.n:
                self.restart = True
            # real time is 10 Hz: do not run faster than the sensor
            time.sleep(max(0.0, 0.1 - (time.perf_counter() - t0)))


def make_handler(runner, index_html):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body=b"", ctype="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, index_html, "text/html; charset=utf-8")
            if u.path == "/frame":
                after = int(q.get("after", ["-1"])[0])
                with runner.lock:
                    buf, seq = runner.latest, runner.seq
                if buf is None or seq <= after:
                    return self._send(204)
                return self._send(200, buf)
            if u.path == "/control":
                with runner.lock:
                    if "policy" in q and q["policy"][0] in ("risk", "distance", "uniform_coarse"):
                        runner.overrides["policy"] = q["policy"][0]; runner.restart = True
                    if "clearing" in q:
                        runner.overrides["ray_clearing"] = q["clearing"][0] == "1"; runner.restart = True
                    if "pause" in q:
                        runner.paused = q["pause"][0] == "1"
                return self._send(200, b"ok", "text/plain")
            self._send(404)
    return H


def record(args, index_html):
    """Run the engine offline and write one self-contained HTML file that replays the result."""
    src = Source(args)
    be = Backend(pick_engine(args), src, args.weights, args.view)
    frames = []
    for i in range(src.n):
        buf = be.step(src.get(i), lod=0.4)
        if i >= args.skip and (i - args.skip) % args.every == 0:
            frames.append(base64.b64encode(gzip.compress(buf, 9)).decode())
        if i % 20 == 0:
            print(f"sweep {i}/{src.n}", flush=True)
    inject = "<script>window.TF_REPLAY_DT=%d;window.TF_REPLAY=%s;</script>" % (int(args.every * 100), json.dumps(frames))
    html = index_html.decode().replace('<script>\n"use strict";', inject + '\n<script>\n"use strict";', 1)
    assert "TF_REPLAY=" in html
    os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    with open(args.record, "w") as f:
        f.write(html)
    print(f"wrote {args.record}: {len(frames)} frames, {os.path.getsize(args.record) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seq", default=None, help="sequence folder in SemanticKITTI layout (contains velodyne/). "
                    "Default: data/gauntlet for the C++ core if it exists, otherwise the built-in simulator")
    ap.add_argument("--engine", choices=("auto", "cpp", "python"), default="auto")
    ap.add_argument("--sensor-height", type=float, default=1.9, help="1.73 for SemanticKITTI")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--view", type=float, default=100.0, help="radius (m) of the map sent to the browser")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "pillarpointnet_sim.npz"))
    ap.add_argument("--record", default=None, help="write a self-contained replay HTML instead of serving")
    ap.add_argument("--every", type=int, default=4, help="record every N-th sweep")
    ap.add_argument("--skip", type=int, default=10)
    args = ap.parse_args()
    gaunt = os.path.join(ROOT, "data", "gauntlet")
    if args.seq is None and args.engine != "python" and os.path.isdir(os.path.join(gaunt, "velodyne")):
        args.seq = gaunt                      # the C++ core's network was trained on this simulator's scenes
    index_html = open(os.path.join(ROOT, "dashboard", "index.html"), "rb").read()
    if args.record:
        return record(args, index_html)
    runner = Runner(args); runner.start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner, index_html))
    print(f"TERRA-FOVEA dashboard: http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
