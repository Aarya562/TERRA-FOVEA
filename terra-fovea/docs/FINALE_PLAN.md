# Before and during the finale

## This week
1. Run it. `python scripts/run_dashboard.py --engine python`, then build the C++ core and run `python run_tests.py`.
   Every team member should be able to start the dashboard alone.
2. Read `docs/HOW_IT_WORKS.md` with the code open. Split ownership: one person each for the grid and policy,
   the network and training, ground and clearing, dashboard and evaluation. Each must be able to explain their part
   without notes.
3. Get one SemanticKITTI sequence running (`docs/REAL_DATA.md`). This is the largest gap between this project
   and your PPT, and the first thing a DRDO jury will probe.
4. Push to your GitHub repository. Do not commit `data/` or `build/`.

## Your PPT promised things this project does not have
Be ready to say "planned, not built" for each, or build it:
- CUDA voxelisation, TensorRT, Jetson deployment, ROS2 node: none exist.
- Validation on SemanticKITTI and RELLIS-3D: not done yet.
- SPVCNN benchmarking: not done.
- "+20% small-object recall", "10x memory", "under 50 ms": the measured figures are in `results/REPORT.md`.
  Quote those and nothing else. 10x against a sparse uniform map holds; 56 ms is the measured time.

## Demo script (5 minutes)
1. Start in Semantic view: road, kerbs, walls, cars, people. Point at the live FPS and the memory panel.
2. Press Resolution: yellow 5 cm near the vehicle, purple 40 cm far away, and finer tiles reaching out
   along the path and around moving objects. The minimap shows the same from above.
3. Press Risk, then switch the policy to Distance only. Detail around distant pedestrians disappears.
   Switch back.
4. Drivable view: show a pit in magenta and the STOP decision, then the gantry (passable) and the boom (blocked).
5. Turn ghost clearing off for ten seconds: trails appear behind moving cars. Turn it on.
6. Run `python run_tests.py` in a terminal: the no-data-loss tests pass in front of them.
7. Close on `results/REPORT.md`: memory table, accuracy by distance, latency.

Keep a recorded replay (`python scripts/run_dashboard.py --record results/replay.html`) on a USB stick in case
the laptop misbehaves.

## Good tasks for the 36 hours
- Real-data numbers and a real-data replay.
- A ROS2 wrapper around the C API, if a mentor asks for integration.
- Path planning on the Drivable layer (A* on the blocked mask) to show the map being used.
- Whatever the mentors ask for in the first review: keep the code small enough that you can change it.
