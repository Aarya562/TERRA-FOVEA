"""Turn results/benchmark.json into charts (results/*.png) and a results table (results/RESULTS.md)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = json.load(open(os.path.join(ROOT, "results", "reference", "benchmark.json")))
V = R["variants"]; B = R["baselines"]
OUT = os.path.join(ROOT, "results", "reference")
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b9b8b1"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#d6d5cf", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlecolor": INK, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlelocation": "left", "grid.color": "#e9e8e3", "savefig.dpi": 160})
NAMES = {"B1_uniform_5cm": "Uniform 5 cm", "B2_uniform_40cm": "Uniform 40 cm", "B3_distance_only": "Distance-only adaptive",
         "B4_terra_fovea": "TERRA-FOVEA (risk-adaptive)"}
ours = V["B4_terra_fovea"]


def mb(v):
    return f"{v / 1000:.2f} GB" if v >= 1000 else (f"{v:.0f} MB" if v >= 100 else f"{v:.1f} MB")


# 1. memory ------------------------------------------------------------------
rows = [("Dense 3D voxel grid, 5 cm (computed)", B["dense_3d_voxel_5cm_mb"]),
        ("Dense 2.5D grid, 5 cm (computed)", B["dense_2p5d_5cm_mb"]),
        ("Uniform 5 cm 2.5D, sparse tiles (measured)", V["B1_uniform_5cm"]["memory_mb_mean"]) if "B1_uniform_5cm" in V else None,
        ("Sparse 3D voxel map, 5 cm (measured)", B["sparse_3d_voxel_5cm_mb"]),
        ("TERRA-FOVEA (measured)", ours["memory_mb_mean"]),
        ("Distance-only adaptive (measured)", V["B3_distance_only"]["memory_mb_mean"]),
        ("Uniform 40 cm 2.5D (measured)", V["B2_uniform_40cm"]["memory_mb_mean"])]
rows = [r for r in rows if r]
fig, ax = plt.subplots(figsize=(8, 3.4))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[BLUE if "TERRA" in r[0] else GREY for r in rows], height=0.62)
ax.set_xscale("log"); ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlabel("map memory, MB (log scale)")
for yy, r in zip(y, rows):
    ax.text(r[1] * 1.12, yy, mb(r[1]), va="center", color=INK, fontsize=9)
ax.set_xlim(1, max(r[1] for r in rows) * 6); ax.grid(axis="x"); ax.set_axisbelow(True)
ax.set_title("Memory to map the same 100 m radius on the same drive")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "memory.png")); plt.close(fig)

# 2. accuracy vs memory --------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
for ax, key, lab in ((axs[0], "inflation_m2", "obstacle over-inflation in the road ahead, m²"),
                     (axs[1], "edge_err_cm", "mean obstacle edge error, cm")):
    for k, col in (("B1_uniform_5cm", GREY), ("B2_uniform_40cm", ORANGE), ("B3_distance_only", AQUA), ("B4_terra_fovea", BLUE)):
        if k not in V:
            continue
        x, yv = V[k]["memory_mb_mean"], V[k]["map"][key]
        ax.scatter([x], [yv], s=70, color=col, zorder=3, edgecolor=SURF, linewidth=2)
        ax.annotate(NAMES[k], (x, yv), xytext=(7, 6), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log"); ax.set_xlabel("map memory, MB (log scale)"); ax.set_ylabel(lab); ax.grid(True); ax.set_axisbelow(True)
    ax.set_ylim(bottom=0); ax.margins(x=0.25, y=0.2)
axs[0].set_title("Lower-left is better: less memory, tighter obstacles")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "accuracy_vs_memory.png")); plt.close(fig)

# 3. classification by distance ------------------------------------------------
bands = ours["seg"]["bands"]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
x = np.arange(len(bands)); wdt = 0.26
for i, (g, col) in enumerate(zip(("terrain", "static obstacle", "dynamic object"), (AQUA, ORANGE, BLUE))):
    vals = [b["group_recall"][i] * 100 for b in bands]
    bars = ax.bar(x + (i - 1) * (wdt + 0.02), vals, wdt, color=col, label=g)
    for xx, v in zip(x + (i - 1) * (wdt + 0.02), vals):
        ax.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, [b["band"] for b in bands]); ax.set_ylim(0, 112); ax.set_ylabel("points correctly grouped, %")
ax.set_xlabel("distance from the sensor"); ax.grid(axis="y"); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=9)
ax.set_title("Classification accuracy by distance (simulated test street)", pad=22)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "accuracy_by_distance.png")); plt.close(fig)

# 4. latency ---------------------------------------------------------------------
st = [("triage", "Geometry triage"), ("network", "Neural segmentation"), ("policy", "Risk plan + regrid"),
      ("fusion", "2.5D fusion"), ("tracking", "Tracking"), ("clearing", "Ghost clearing")]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
vals = [ours["timing_ms"][k] for k, _ in st]; y = np.arange(len(st))[::-1]
ax.barh(y, vals, color=BLUE, height=0.6)
for yy, v in zip(y, vals):
    ax.text(v + 0.6, yy, f"{v:.1f} ms", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [n for _, n in st]); ax.set_xlabel("mean time per sweep, ms"); ax.grid(axis="x"); ax.set_axisbelow(True)
ax.set_xlim(0, max(vals) * 1.25)
ax.set_title(f"Where the {ours['timing_ms']['total']:.0f} ms per sweep goes ({ours['fps']:.1f} FPS, 2 CPU cores, no GPU)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "latency.png")); plt.close(fig)

# 5. ablations ---------------------------------------------------------------------
if "A1_no_ray_clearing" in V:
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.0))
    a = V["A1_no_ray_clearing"]
    axs[0].bar(["clearing off", "clearing on"], [a["map"]["phantom_m2"], ours["map"]["phantom_m2"]], color=[GREY, BLUE], width=0.5)
    for i, v in enumerate([a["map"]["phantom_m2"], ours["map"]["phantom_m2"]]):
        axs[0].text(i, v, f"{v:.1f} m²", ha="center", va="bottom", fontsize=9, color=INK)
    axs[0].set_ylabel("phantom obstacle area in the road, m²"); axs[0].set_title("Ghost clearing"); axs[0].grid(axis="y"); axs[0].set_axisbelow(True)
    f_ = V["A2_ai_on_all_points"]
    axs[1].bar(["network on every point", "selective (ours)"], [f_["timing_ms"]["network"], ours["timing_ms"]["network"]], color=[GREY, BLUE], width=0.5)
    for i, (v, acc) in enumerate([(f_["timing_ms"]["network"], f_["seg"]["overall"]["acc"]), (ours["timing_ms"]["network"], ours["seg"]["overall"]["acc"])]):
        axs[1].text(i, v, f"{v:.0f} ms, accuracy {acc * 100:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    axs[1].set_ylabel("network time per sweep, ms"); axs[1].set_title("Selective AI"); axs[1].grid(axis="y"); axs[1].set_axisbelow(True)
    for ax in axs:
        ax.margins(y=0.18)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ablations.png")); plt.close(fig)

# results table ---------------------------------------------------------------------
L = ["# Measured results", "",
     f"Scene `{R['scene']}`, {R['frames']} sweeps at 10 Hz, about {ours['points_mean']:.0f} returns per sweep. "
     "Every row saw the same sweeps. Timing is from the machine that ran the benchmark (CPU only).", "",
     "| Variant | Memory | FPS | Over-inflation | Phantom area | Edge error | Point accuracy | mIoU |",
     "|---|---|---|---|---|---|---|---|"]
for k, v in V.items():
    L.append(f"| {NAMES.get(k, k)} | {mb(v['memory_mb_mean'])} | {v['fps']:.1f} | {v['map']['inflation_m2']:.2f} m² | "
             f"{v['map']['phantom_m2']:.2f} m² | {v['map']['edge_err_cm']:.1f} cm | {v['seg']['overall']['acc'] * 100:.1f}% | {v['seg']['overall']['miou'] * 100:.1f}% |")
L += ["", "## Memory baselines", "", "| Representation | Memory | vs TERRA-FOVEA |", "|---|---|---|"]
for n, val in rows:
    L.append(f"| {n} | {mb(val)} | {val / ours['memory_mb_mean']:.1f}x |")
L += ["", "## TERRA-FOVEA by distance", "", "| Band | Points | Accuracy | mIoU | Terrain | Static | Dynamic |", "|---|---|---|---|---|---|---|"]
for b in bands:
    g = b["group_recall"]
    L.append(f"| {b['band']} | {b['points']:,} | {b['acc'] * 100:.1f}% | {b['miou'] * 100:.1f}% | {g[0] * 100:.1f}% | {g[1] * 100:.1f}% | {g[2] * 100:.1f}% |")
L += ["", "Per-class IoU: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in ours["seg"]["overall"]["iou"].items()), "",
      "## Moving objects", "",
      f"Recall {ours['dynamic']['recall'] * 100:.0f}%, precision {ours['dynamic']['precision'] * 100:.0f}% over {ours['dynamic']['truth']} object-sweeps within 50 m; "
      f"mean speed error {ours['dynamic']['speed_err_ms']:.2f} m/s.", "",
      "## Latency (TERRA-FOVEA)", "", "| Stage | Mean | 95th percentile |", "|---|---|---|"]
for k, n in st + [("total", "Total")]:
    L.append(f"| {n} | {ours['timing_ms'][k]:.1f} ms | {ours['timing_ms_p95'][k]:.1f} ms |")
if "robustness" in R:
    r = R["robustness"]
    L += ["", "## Robustness: 30% extra dropout and +3 cm range noise", "",
          f"Accuracy {r['seg']['overall']['acc'] * 100:.1f}% (clean {ours['seg']['overall']['acc'] * 100:.1f}%), mIoU {r['seg']['overall']['miou'] * 100:.1f}%, "
          f"over-inflation {r['map']['inflation_m2']:.2f} m², phantom area {r['map']['phantom_m2']:.2f} m², "
          f"moving-object recall {r['dynamic']['recall'] * 100:.0f}%."]
L += ["", "## What the columns mean", "",
      "- Over-inflation: road area the map marks blocked that is actually free, within 0.6 m of a real obstacle. Coarse cells cause it.",
      "- Phantom area: road area marked blocked with no real obstacle within 0.6 m. Ghost trails and false alarms cause it.",
      "- Edge error: mean distance from a blocked map pixel to the nearest truly blocked pixel.",
      "- All three are measured on a 5 cm raster of the road from 3 m to 60 m ahead of the vehicle, against the simulator's exact geometry."]
open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
print("wrote charts and RESULTS.md to", OUT)
