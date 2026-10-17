"""Summary figure for the report (static PNG, light surface)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common as C

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2 = "#2a78d6", "#eb6834"

st = json.load(open(os.path.join(C.RESULTS, "stage2_tests.json")))
ex = json.load(open(os.path.join(C.RESULTS, "exploratory_E1_E2_E3.json")))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=SURF,
                         gridspec_kw={"width_ratios": [1, 1, 1.25], "wspace": 0.55})


def style(ax, title, sub, xlabel):
    ax.set_facecolor(SURF)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=34)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_xlabel(xlabel, fontsize=8.5)


def hbars(ax, labels, vals, los, his, color, fmt="{:.2f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.62, color=color, edgecolor=SURF, linewidth=2)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor=INK, elinewidth=1, capsize=0)
    for yy, v, h in zip(y, vals, his):
        ax.text(h + 0.015, yy, fmt.format(v), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color=INK)


# A: synthetic shapes (exploratory sample incl. ellipse)
E1 = ex["E1_synthetic"]
order = [("V", "V (sharp apex)"), ("U", "U (rounded apex)"), ("E", "Ellipse (no corners)"),
         ("V_flip", "V, apex up"), ("O", "Disk (equal area)")]
ax = axes[0]
hbars(ax, [l for _, l in order], [E1[k]["mean"] for k, _ in order], [E1[k]["lo"] for k, _ in order],
      [E1[k]["hi"] for k, _ in order], S1)
ax.axvline(0.30, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(0.305, 4.45, "pre-registered\n'strong' = 0.30", fontsize=7.5, color=INK2, va="top")
ax.set_xlim(0, 0.45)
style(ax, "A  Shape alone barely drives it", "Textured shapes on gray, no category content\n(exploratory, n=120 each)",
      "Drive (fraction of natural top-image max)")

# B: silhouette vs interior (pre-registered F3)
F3 = st["F3"]["all"]
order = [("bg_flat", "Background flattened"), ("bg_scr", "Background scrambled"),
         ("sil_flat", "Outline kept, interior flat"), ("sil_scr", "Outline kept, interior scrambled")]
ax = axes[1]
hbars(ax, [l for _, l in order], [F3[k]["R_mean"]["mean"] for k, _ in order],
      [F3[k]["R_mean"]["lo"] for k, _ in order], [F3[k]["R_mean"]["hi"] for k, _ in order], S1)
ax.set_xlim(0, 1.2)
style(ax, "B  The interior carries the drive", "Unit's top 200 natural images, edited\n(pre-registered F3)",
      "Retained fraction of original drive")

# C: apertures on cat faces, 1x vs 4x area (exploratory)
E2 = ex["E2_apertures"]
keys = [("V", "V"), ("U", "U"), ("E", "Ellipse"), ("O", "Disk")]
ax = axes[2]
y = np.arange(len(keys))[::-1]
for j, (sc, col, lab) in enumerate([("area_x1", S1, "Aperture 1× (pre-registered size)"),
                                    ("area_x4", S2, "Aperture 4× (shows most of the face)")]):
    d = E2[sc]["cat_faces"]
    v = np.array([d[k]["mean"] for k, _ in keys]); lo = np.array([d[k]["lo"] for k, _ in keys])
    hi = np.array([d[k]["hi"] for k, _ in keys])
    yy = y + (0.19 if j == 0 else -0.19)
    ax.barh(yy, v, height=0.36, color=col, edgecolor=SURF, linewidth=2, label=lab)
    ax.errorbar(v, yy, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1)
    for a, b, c in zip(yy, v, hi):
        ax.text(c + 0.012, a, f"{b:.2f}", va="center", fontsize=8, color=INK)
full = E2["area_x1"]["cat_faces"]["full"]["mean"]
ax.axvline(full, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(full - 0.01, 3.62, f"whole face\nno aperture: {full:.2f}", fontsize=7.5, color=INK2, ha="right", va="top")
ax.set_yticks(y); ax.set_yticklabels([l for _, l in keys], fontsize=9, color=INK)
ax.set_xlim(0, 0.75)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), frameon=False, fontsize=8, labelcolor=INK, ncol=2)
style(ax, "C  Once the face is visible, the disk wins", "Oxford Pets cat faces through equal-area apertures\n(exploratory, n=200)",
      "Drive (fraction of natural top-image max)")

fig.text(0.01, -0.12, "InceptionV1 mixed4e:55. Bars: means; whiskers: 95% bootstrap CIs. "
         "Normalization: mean drive of the unit's top 0.2% ImageNetV2 images = 1.0.", fontsize=7.5, color=INK2)
fig.savefig(os.path.join(C.RESULTS, "fig_summary.png"), dpi=180, bbox_inches="tight", facecolor=SURF)
print("saved")
