"""Exploratory (NOT pre-registered) follow-ups for unit 55, run after the pre-registered results were seen.

E1  Ellipse control: is the V/U > O effect about elongation rather than a pointed apex?
    Adds an ellipse with the V's area and the V's long-axis extent.
E2  Large apertures (4x area): does the aperture-outline effect survive when most of the face is visible?
E3  Apex orientation: V at theta* vs flipped 180 degrees (apex up), same shape otherwise.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import common as C


def u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def ellipse_polygon(width, area, theta, center=(112, 112)):
    a = width / 2
    b = area / (np.pi * a)
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    p = np.stack([a * np.cos(ang), b * np.sin(ang)], 1) + np.asarray(center)
    return C._rot(p, theta, np.asarray(center))


def head_canvas(d, scale_to):
    img = C.load_rgb(d["path"])
    x0, y0, x1, y1 = d["head"]
    hs = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = scale_to / hs
    w, h = img.size
    im = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BICUBIC)
    can = Image.new("RGB", (224, 224), tuple(int(v * 255) for v in C.GRAY))
    can.paste(im, (round(112 - cx * k), round(112 - cy * k)))
    return C.to01(can)


def translate_to_center(img01, iy, ix):
    py, px = 16 * iy + 8, 16 * ix + 8
    can = C.gray_canvas()
    dy, dx = 112 - py, 112 - px
    ys, xs = slice(max(0, dy), min(224, 224 + dy)), slice(max(0, dx), min(224, 224 + dx))
    yo, xo = slice(max(0, -dy), min(224, 224 - dy)), slice(max(0, -dx), min(224, 224 - dx))
    can[ys, xs] = img01[yo, xo]
    return can


def mci(x):
    m, lo, hi = C.boot_ci(np.asarray(x, float))
    return {"mean": m, "lo": lo, "hi": hi, "n": len(x)}


def pdiff(a, b):
    m, lo, hi = C.boot_ci(np.asarray(a, float) - np.asarray(b, float))
    return {"diff": m, "lo": lo, "hi": hi}


def main():
    rng = np.random.default_rng(11)
    probe = C.Probe()
    A = float(np.load(os.path.join(C.WORK, "aref.npz"))["aref4e"][C.UNIT])
    st1 = json.load(open(os.path.join(C.RESULTS, "stage1_tuning.json")))
    TH, AL, SS_ = st1["theta_star"], st1["alpha_star"], st1["s_star"]
    area = C.polygon_area("V", AL, SS_)
    vpoly = C.shape_polygon("V", AL, SS_, 0)
    width = float(vpoly[:, 0].max() - vpoly[:, 0].min())
    dtd = [p for c in C.DTD_STAGE2 for p in C.dtd_paths([c])[c]]
    out = {"note": "EXPLORATORY, not pre-registered; run after seeing pre-registered results",
           "ellipse_width_px": width, "area_px2": area}

    # ---------------- E1 + E3: synthetic, neutral texture, gray, no eyes
    stims, meta = [], []
    for shape in ["V", "U", "O", "E", "V_flip"]:
        for _ in range(120):
            th = TH + rng.choice([-15, 0, 15])
            s = float(rng.choice([SS_, 0.8 * SS_]))
            ar_ = area * (s / SS_) ** 2
            ctr = (112 + rng.uniform(-8, 8), 112 + rng.uniform(-8, 8))
            if shape == "E":
                poly = ellipse_polygon(width * s / SS_, ar_, th, ctr)
            elif shape == "V_flip":
                poly = C.shape_polygon("V", AL, s, th + 180, center=ctr, area=ar_)
            else:
                poly = C.shape_polygon(shape, AL, s, th, center=ctr, area=ar_)
            tex = C.random_texture(dtd[rng.integers(len(dtd))], rng)
            stims.append(u8(C.composite(C.gray_canvas(), tex, C.fill_mask(poly)))); meta.append(shape)
    rel = C.run_arrays(probe, stims, keys={"drive"})["drive"] / A
    meta = np.array(meta)
    out["E1_synthetic"] = {s: mci(rel[meta == s]) for s in ["V", "U", "O", "E", "V_flip"]}
    out["E1_V_minus_E"] = C.boot_diff(rel[meta == "V"], rel[meta == "E"])
    out["E1_E_minus_O"] = C.boot_diff(rel[meta == "E"], rel[meta == "O"])
    out["E3_V_minus_Vflip"] = C.boot_diff(rel[meta == "V"], rel[meta == "V_flip"])

    # ---------------- E2: apertures at 1x and 4x area, with ellipse
    pets = C.pets_index()
    ids = [i for i, d in enumerate(pets) if d["species"] == "cat" and d["head"] is not None and
           max(d["head"][2] - d["head"][0], d["head"][3] - d["head"][1]) >= 40]
    rng.shuffle(ids)
    S = np.load(os.path.join(C.WORK, f"scan_unit{C.UNIT}.npz"))
    imn_rel, imn_cls = S["imn_drive"] / A, S["imn_cls"]
    cats = C.imagenet_categories()
    veh = set(C.resolve_names(C.VEHICLES, cats, []))
    order = np.argsort(-imn_rel)[:100]
    top_anim = [i for i in order if imn_cls[i] < 398]
    top_veh = [i for i in order if imn_cls[i] in veh]
    def tr(i):
        iy, ix = S["imn_argmax_yx"][i]
        return translate_to_center(C.to01(C.center_crop_224(C.load_rgb(S["imn_paths"][i]))[0]), iy, ix)
    contents = {"cat_faces": [head_canvas(pets[i], SS_) for i in ids[:200]],
                "top_animal": [tr(i) for i in top_anim], "top_vehicle": [tr(i) for i in top_veh],
                "neutral": [C.random_texture(dtd[rng.integers(len(dtd))], rng) for _ in range(150)]}
    E2 = {}
    for scale in [1, 4]:
        ar_ = area * scale
        k = np.sqrt(scale)
        masks = {"V": C.fill_mask(C.shape_polygon("V", AL, SS_ * k, TH, area=ar_)),
                 "U": C.fill_mask(C.shape_polygon("U", AL, SS_ * k, TH, area=ar_)),
                 "O": C.fill_mask(C.shape_polygon("O", None, SS_ * k, TH, area=ar_)),
                 "E": C.fill_mask(ellipse_polygon(width * k, ar_, TH))}
        res = {}
        for name, lst in contents.items():
            imgs, mm = [], []
            for img in lst:
                imgs.append(u8(img)); mm.append("full")
                for a, m in masks.items():
                    imgs.append(u8(C.composite(C.gray_canvas(), img, m))); mm.append(a)
            r = C.run_arrays(probe, imgs, keys={"drive"})["drive"] / A
            mm = np.array(mm)
            t = {a: r[mm == a] for a in ["full", "V", "U", "O", "E"]}
            res[name] = {a: mci(t[a]) for a in t}
            res[name]["V_minus_O"] = pdiff(t["V"], t["O"])
            res[name]["V_minus_E"] = pdiff(t["V"], t["E"])
            res[name]["E_minus_O"] = pdiff(t["E"], t["O"])
            res[name]["frac_of_full_V"] = float(t["V"].mean() / t["full"].mean())
        E2[f"area_x{scale}"] = res
    out["E2_apertures"] = E2
    C.save_json(out, "exploratory_E1_E2_E3.json")
    r3 = lambda x: round(x, 3)
    print("E1", {k: r3(v["mean"]) for k, v in out["E1_synthetic"].items()}, "V-E", [r3(x) for x in out["E1_V_minus_E"]],
          "E-O", [r3(x) for x in out["E1_E_minus_O"]], "V-Vflip", [r3(x) for x in out["E3_V_minus_Vflip"]])
    for sc, res in E2.items():
        for name, v in res.items():
            print(sc, name, {a: r3(v[a]["mean"]) for a in ["full", "V", "U", "O", "E"]},
                  "V-O", r3(v["V_minus_O"]["diff"]), "V-E", r3(v["V_minus_E"]["diff"]), "E-O", r3(v["E_minus_O"]["diff"]))


if __name__ == "__main__":
    main()
