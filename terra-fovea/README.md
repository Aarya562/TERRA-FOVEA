# TERRA-FOVEA

Risk-adaptive, variable-resolution 2.5D LiDAR mapping. SIH 2026, problem statement 26053, Team Veyronix.

A LiDAR sweep goes in. Out comes a 2.5D elevation map with semantic layers whose cells are 5 cm
where it matters and up to 40 cm where it does not, plus tracked moving objects and a
CLEAR / CAUTION / STOP decision for the path ahead.

**Everything measured here is on simulated LiDAR.** No real dataset has been run yet. `docs/REAL_DATA.md`
is the procedure for SemanticKITTI; do that before the finale.

## What is in the box

| Folder | What it is |
|---|---|
| `core/` | The real-time engine in dependency-free C++17 (1,100 lines) with a plain C API. This is the product. |
| `python/` | ctypes binding, simulator, trainer and evaluation harness for the C++ engine. |
| `terrafovea/` | An independent reference implementation of the same pipeline in readable NumPy, with its own simulator. Use it to understand the algorithms and to run without a compiler. |
| `dashboard/` | One offline WebGL dashboard that shows either engine live. |
| `scripts/` | Dashboard server, reference benchmark, training (NumPy and PyTorch). |
| `tests/`, `run_tests.py` | 14 Python tests plus the 9 C++ self-tests. |
| `results/` | `REPORT.md` (C++ engine) and `reference/RESULTS.md` (NumPy reference), with charts. |
| `docs/` | How it works, judge Q&A, real-data guide, finale plan. |

## Run it

Python 3.9+ with `pip install numpy scipy matplotlib`.

```
python scripts/run_dashboard.py --engine python      # works immediately, no compiler
```
Open http://localhost:8765. This runs the NumPy reference on its built-in simulated street.

For full speed, build the C++ core and generate its test drive:

```
./build.sh                    # Linux / macOS / WSL.   Windows: build.bat   (needs g++ or MSVC)
python python/sim.py          # writes data/gauntlet and training scenes (about 1.7 GB, a few minutes)
python scripts/run_dashboard.py
python run_tests.py
```

`build.bat` and `CMakeLists.txt` have not been tried on Windows. If the build fails there, use WSL, or
stay on `--engine python` for the demo.

Dashboard controls: drag to orbit, wheel to zoom, keys 1-6 switch the colouring. The policy buttons
restart the map under distance-only or uniform 40 cm so the judges can watch memory and detail change.

## Reproduce the numbers

```
python python/evaluate.py                 # C++ engine  -> results/REPORT.md
python scripts/run_benchmark.py           # reference   -> results/reference/benchmark.json  (10 min)
python scripts/make_report.py             #             -> results/reference/RESULTS.md + charts
```

## Headline results (simulated, C++ engine, 2 CPU cores, no GPU)

| | |
|---|---|
| Speed | 56 ms per 84,000-point sweep, 17.8 FPS |
| Map memory | 7.8 MB, against 77.7 MB for the same engine at uniform 5 cm (9.9x) and 1.22 GB for a dense 5 cm voxel grid (156x) |
| Segmentation | mIoU 0.947 with the network run on 59% of the points |
| Ghost trails | 0.46 m² of stale obstacle with ray clearing, 16.5 m² without |
| Pits | flagged at 6 to 10 m range; overhead clearance measured within 2 cm |

Full tables: `results/REPORT.md`. The NumPy reference, on a different simulated street, shows the same
ordering between uniform, distance-only and risk-adaptive maps: `results/reference/RESULTS.md`.

## Known limits

- Simulated data only so far. Accuracy on real LiDAR will be lower until the network is trained on real data.
- Poses are taken from the dataset. There is no SLAM or odometry in this project.
- The ground model assumes terrain is locally planar over a few metres. Steep, broken off-road ground needs a stronger ground segmenter.
- Pits are only visible at short range (grazing angle), so the vehicle must be slow enough to stop within about 6 m.
- The two engines were written separately and their numbers differ in detail. Quote the C++ report as the system's performance.
- The NumPy reference runs at about 6 FPS. It is for understanding and checking, not for real time.
