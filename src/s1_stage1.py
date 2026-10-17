"""Stage 1: synthetic shape tuning battery -> validity check V1 and frozen theta*, alpha*, s*."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import common as C

ALPHAS = [30, 60, 90, 120]
THETAS = list(range(0, 360, 15))
SIZES = [64, 112, 160]
STROKE = 6


def bar_polygon(size, theta, width_frac=0.3):
    w, h = width_frac * size, size
    p = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) + C.CANVAS / 2
    return C._rot(p, theta, np.array([C.CANVAS / 2, C.CANVAS / 2]))


def u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def make_stage1(rng):
    tex = C.dtd_paths(C.DTD_STAGE1)
    stims, meta = [], []
    light, dark = np.full((224, 224, 3), 0.85, np.float32), np.full((224, 224, 3), 0.15, np.float32)
    shapes = [("V", a) for a in ALPHAS] + [("U", a) for a in ALPHAS] + [("O", None), ("bar", None)]
    for (shape, a), th, s in itertools.product(shapes, THETAS, SIZES):
        # outlines, both polarities
        line = C.outline_polyline(shape, a or 60, s, th)
        sm = C.stroke_mask(line, STROKE, closed=False)
        stims.append(u8(C.composite(light, dark, sm))); meta.append((shape, a, th, s, "outline_dark"))
        stims.append(u8(C.composite(dark, light, sm))); meta.append((shape, a, th, s, "outline_light"))
        # filled textured silhouettes on gray
        if shape == "bar":
            poly = bar_polygon(s, th)
        elif shape == "O":
            poly = C.shape_polygon("O", None, s, th)
        else:
            poly = C.shape_polygon(shape, a, s, th, area=C.polygon_area("V", a, s))
        m = C.fill_mask(poly)
        for cat in C.DTD_STAGE1:
            for _ in range(2):
                t = C.random_texture(tex[cat][rng.integers(len(tex[cat]))], rng)
                stims.append(u8(C.composite(C.gray_canvas(), t, m))); meta.append((shape, a, th, s, "filled_" + cat))
    return stims, meta


def make_arcs():
    """Thin dark arcs (90 deg span) with their midpoint at the canvas centre, for curve-detector check."""
    out, meta = [], []
    bg = np.full((224, 224, 3), 0.85, np.float32)
    for r, th in itertools.product([20, 40], THETAS):
        ang = np.deg2rad(np.linspace(-45, 45, 60))
        pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)   # arc around origin, midpoint at (r,0)
        pts = pts - np.array([r, 0.0]) + C.CANVAS / 2           # midpoint at centre
        pts = C._rot(pts, th, np.array([C.CANVAS / 2, C.CANVAS / 2]))
        out.append(C.composite(bg, np.zeros_like(bg) + 0.05, C.stroke_mask(pts, 3)))
        meta.append((r, th))
    return out, meta


def main():
    rng = np.random.default_rng(1)
    probe = C.Probe()
    ar = np.load(os.path.join(C.WORK, "aref.npz"))
    aref4e, aref3b = ar["aref4e"], ar["aref3b"]

    stims, meta = make_stage1(rng)
    print("stage-1 stimuli:", len(stims), flush=True)
    r = C.run_arrays(probe, stims, keys={"drive", "max4e", "pre_mean"})
    rel = r["drive"] / aref4e[C.UNIT]
    pop = r["max4e"] / aref4e[None, :]

    arcs, ameta = make_arcs()
    ra = C.run_arrays(probe, arcs, keys={"max3b", "center3b"})

    # V1a: fraction of mixed4e channels driven to >= 0.3 of their own A_ref by some stage-1 stimulus
    frac_driven = float((pop.max(0) >= 0.3).mean())
    # V1b: curve detectors' best response to arcs
    curve_max = {int(u): float(ra["max3b"][:, u].max() / aref3b[u]) for u in C.CURVE_UNITS_3B}
    curve_center = {int(u): float(ra["center3b"][:, u].max() / aref3b[u]) for u in C.CURVE_UNITS_3B}
    curve_median = float(np.median(list(curve_max.values())))

    # theta*, alpha*, s*: argmax of mean rel over filled V silhouettes
    shape = np.array([m[0] for m in meta]); alpha = np.array([m[1] if m[1] else 0 for m in meta])
    theta = np.array([m[2] for m in meta]); size = np.array([m[3] for m in meta])
    rend = np.array([m[4] for m in meta])
    filled = np.char.startswith(rend.astype(str), "filled")
    best, table = None, {}
    for a, th, s in itertools.product(ALPHAS, THETAS, SIZES):
        sel = (shape == "V") & filled & (alpha == a) & (theta == th) & (size == s)
        v = float(rel[sel].mean())
        table[f"{a}_{th}_{s}"] = v
        if best is None or v > best[0]:
            best = (v, a, th, s)

    def summary(sel):
        return {"n": int(sel.sum()), "mean_rel": float(rel[sel].mean()), "max_rel": float(rel[sel].max())}

    by_shape = {}
    for sh in ["V", "U", "O", "bar"]:
        by_shape[sh] = {"filled": summary((shape == sh) & filled),
                        "outline": summary((shape == sh) & ~filled)}
    # tuning curve of filled V at alpha*, s*
    tune = {int(th): table[f"{best[1]}_{th}_{best[3]}"] for th in THETAS}

    out = {
        "theta_star": best[2], "alpha_star": best[1], "s_star": best[3], "best_mean_rel": best[0],
        "V1a_frac_mixed4e_channels_driven_ge_0.3": frac_driven,
        "V1b_curve_detectors_max_rel": curve_max, "V1b_curve_detectors_center_rel": curve_center,
        "V1b_median": curve_median,
        "V1_pass": bool(frac_driven >= 0.10 and curve_median >= 0.5),
        "stage1_by_shape_exploratory": by_shape,
        "filled_V_orientation_tuning_at_alpha_s_star": tune,
        "unit55_max_rel_any_stage1": float(rel.max()),
        "unit55_best_stimulus": list(map(str, meta[int(rel.argmax())])),
        "n_stimuli": len(stims),
    }
    C.save_json(out, "stage1_tuning.json")
    np.savez(os.path.join(C.WORK, "stage1.npz"), rel=rel, pop=pop, shape=shape, alpha=alpha, theta=theta,
             size=size, rend=rend.astype(str))
    print({k: v for k, v in out.items() if not isinstance(v, dict)})

    # example sheet: top-8 and a fixed set of stage-1 stimuli for the record
    sheet = Image.new("RGB", (8 * 112, 112))
    for j, i in enumerate(np.argsort(-rel)[:8]):
        sheet.paste(Image.fromarray(stims[i]).resize((112, 112)), (j * 112, 0))
    sheet.save(os.path.join(C.RESULTS, "fig_stage1_top8_stimuli.jpg"), quality=88)


if __name__ == "__main__":
    main()
