"""Study 2, stages 0-1 for a mixed4d pool-branch unit: rescan, V0 (identity), stage-1 tuning, V1."""
import os, sys, argparse, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import common as C
import unitprobe as U
import s1_stage1 as S1   # reuse the identical stage-1 battery (same seed)

ap = argparse.ArgumentParser()
ap.add_argument("--unit", type=int, default=493)
args = ap.parse_args()
OUT = os.path.join(C.PROJ, f"results_4d{args.unit}")
os.makedirs(OUT, exist_ok=True)
MAMMALS = set(range(151, 300)) | set(range(330, 389))


def save(obj, name):
    import json
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_crop(path):
    img, _ = C.center_crop_224(C.load_rgb(path))
    return (C.to01(img) * 255).astype(np.uint8)


def scan(probe, paths):
    outs = {}
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(paths), 64):
            o = U.run_arrays(probe, list(ex.map(load_crop, paths[s:s + 64])))
            for k, v in o.items():
                outs.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in outs.items()}


def main():
    t0 = time.time()
    probe = U.UnitProbe(args.unit)
    cats = C.imagenet_categories()
    ar = np.load(os.path.join(C.WORK, "aref.npz"))
    aref4d = ar["aref4d"]
    A = float(aref4d[args.unit])

    s0 = np.load(os.path.join(C.WORK, "scan_imnv2.npz")); p0 = np.load(os.path.join(C.WORK, "scan_pets.npz"))
    ri = scan(probe, list(s0["paths"])); rp = scan(probe, list(p0["paths"]))
    out = {"imn_" + k: v for k, v in ri.items()}
    out.update({"pets_" + k: v for k, v in rp.items()})
    out.update(imn_cls=s0["cls"], imn_paths=s0["paths"], pets_paths=p0["paths"], pets_breed=p0["breed"],
               pets_species=p0["species"])
    np.savez(os.path.join(C.WORK, f"scan4d_unit{args.unit}.npz"), **out)
    # exactness checks
    recon = ri["route_at"] @ probe.w + probe.b
    decomp_err = float(np.abs(recon - ri["pre_at"]).max())
    relu_err = float(max(ri["post_equals_relu_pre_err"].max(), rp["post_equals_relu_pre_err"].max()))
    # A_ref consistency with study-1 scan (same readout: spatial max of mixed4d channel)
    k = int(round(0.002 * len(ri["drive"])))
    A_here = float(np.sort(ri["drive"])[-k:].mean())

    order = np.argsort(-ri["drive"])
    top50 = order[:50]
    n_mamm = int(sum(int(s0["cls"][i]) in MAMMALS for i in top50))
    v0 = {"unit": args.unit, "A_ref": A, "A_ref_recomputed": A_here,
          "top50": [{"rank": j + 1, "class": cats[int(s0["cls"][i])], "rel": float(ri["drive"][i] / A)}
                    for j, i in enumerate(top50)],
          "n_mammal_in_top50": n_mamm, "pass": bool(n_mamm >= 25),
          "criterion": ">=50% of top-50 ImageNetV2 images in mammal classes (151-299, 330-388)",
          "decomposition_max_abs_error": decomp_err, "post_vs_relu_pre_max_err": relu_err,
          "pre_at_scale": float(np.abs(ri["pre_at"]).max())}
    save(v0, "v0_unit_identity.json")
    print("V0", v0["pass"], n_mamm, "decomp err", decomp_err, "relu err", relu_err, f"A {A:.1f} vs {A_here:.1f}", flush=True)
    sheet = Image.new("RGB", (6 * 112, 4 * 112))
    for j, i in enumerate(order[:24]):
        sheet.paste(Image.fromarray(load_crop(s0["paths"][i])).resize((112, 112)), ((j % 6) * 112, (j // 6) * 112))
    sheet.save(os.path.join(OUT, "fig_v0_top24_imagenetv2.jpg"), quality=88)

    # stage 1: identical battery to study 1
    stims, meta = S1.make_stage1(np.random.default_rng(1))
    r = U.run_arrays(probe, stims, keys={"drive", "max_pop"})
    rel = r["drive"] / A
    pop = r["max_pop"] / aref4d[None, :]
    frac = float((pop.max(0) >= 0.3).mean())
    shape = np.array([m[0] for m in meta]); alpha = np.array([m[1] if m[1] else 0 for m in meta])
    theta = np.array([m[2] for m in meta]); size = np.array([m[3] for m in meta])
    filled = np.char.startswith(np.array([m[4] for m in meta]).astype(str), "filled")
    best, table = None, {}
    for a, th, s in itertools.product(S1.ALPHAS, S1.THETAS, S1.SIZES):
        v = float(rel[(shape == "V") & filled & (alpha == a) & (theta == th) & (size == s)].mean())
        table[f"{a}_{th}_{s}"] = v
        if best is None or v > best[0]:
            best = (v, a, th, s)
    by_shape = {sh: {"filled_mean": float(rel[(shape == sh) & filled].mean()),
                     "outline_mean": float(rel[(shape == sh) & ~filled].mean()),
                     "max": float(rel[shape == sh].max())} for sh in ["V", "U", "O", "bar"]}
    st1 = {"theta_star": best[2], "alpha_star": best[1], "s_star": best[3], "best_mean_rel": best[0],
           "V1_frac_mixed4d_channels_driven_ge_0.3": frac, "V1_pass": bool(frac >= 0.10),
           "stage1_by_shape_exploratory": by_shape,
           "filled_V_orientation_tuning": {int(th): table[f"{best[1]}_{th}_{best[3]}"] for th in S1.THETAS},
           "unit_max_rel_any_stage1": float(rel.max()), "best_stimulus": list(map(str, meta[int(rel.argmax())]))}
    save(st1, "stage1_tuning.json")
    print({k: v for k, v in st1.items() if not isinstance(v, dict)}, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
