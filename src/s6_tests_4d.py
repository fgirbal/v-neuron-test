"""Study 2, stage 2: pre-registered tests for a mixed4d pool-branch unit (4d:493).

Usage: python -I s6_tests_4d.py --unit 493                     (the reported run)
       python -I s6_tests_4d.py --unit 492 --tag _debug492     (pipeline debugging on another unit)
Stage-1 parameters always come from results_4d493/stage1_tuning.json (frozen for the study unit).
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import common as C
import unitprobe as U

ap = argparse.ArgumentParser()
ap.add_argument("--unit", type=int, default=493)
ap.add_argument("--tag", default="")
args = ap.parse_args()
OUT = os.path.join(C.PROJ, "results_4d493")
MAMMALS = set(range(151, 300)) | set(range(330, 389))
LOG = []


def log(m):
    print(m, flush=True); LOG.append(m)


def u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def save(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ------------------------------------------------------------------ stats helpers
def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    m, lo, hi = C.boot_ci(x)
    return {"n": int(len(x)), "mean": m, "lo": lo, "hi": hi}


def diff_ci(a, b, paired=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        m, lo, hi = C.boot_ci(a - b); _, lo9, hi9 = C.boot_ci(a - b, alpha=0.10)
    else:
        m, lo, hi = C.boot_diff(a, b); _, lo9, hi9 = C.boot_diff(a, b, alpha=0.10)
    return {"diff": m, "lo": lo, "hi": hi, "lo90": lo9, "hi90": hi9}


def approx(d):          # "≈": 90% CI inside ±0.10
    return bool(d["lo90"] > -0.10 and d["hi90"] < 0.10)


def big(d, t=0.10):     # difference ≥ t with 95% CI lower bound > 0
    return bool(d["diff"] >= t and d["lo"] > 0)

# ------------------------------------------------------------------ geometry
def ellipse_like_V(alpha, area, theta, center=(112, 112)):
    """D1: same area as the V; long axis = V's longest extent, aligned with it."""
    v0 = C.shape_polygon("V", alpha, 100, 0, center=(0, 0), area=area)
    ex, ey = np.ptp(v0[:, 0]), np.ptp(v0[:, 1])
    if ey >= ex:
        sy = ey / 2; sx = area / (np.pi * sy)
    else:
        sx = ex / 2; sy = area / (np.pi * sx)
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    p = np.stack([sx * np.cos(ang), sy * np.sin(ang)], 1) + np.asarray(center)
    return C._rot(p, theta, np.asarray(center))


def u_poly_study2(alpha, area, theta, center=(112, 112), t_frac=0.5):
    """D2: rounded-apex U whose arc is tangent to the arms at t_frac of their length (valid at any angle).
    common.u_polygon (radius 0.35*L) is malformed for narrow angles such as 30 degrees."""
    a = np.deg2rad(alpha) / 2
    tri, L = C.v_polygon(alpha, 100)
    t = t_frac * L; r = t * np.tan(a); d = t / np.cos(a)
    angs = np.linspace(np.pi - a, a, 40)
    arc = np.stack([r * np.cos(angs), -d + r * np.sin(angs)], 1)
    p = np.concatenate([[tri[0]], arc, [tri[1]]], 0)
    A0, c0 = C._polygon_area_centroid(p)
    p = (p - c0) * np.sqrt(area / A0) + np.asarray(center)
    return C._rot(p, theta, np.asarray(center))


def poly_for(shape, alpha, s, theta, area, center=(112, 112)):
    if shape == "E":
        return ellipse_like_V(alpha, area, theta, center)
    if shape == "U":
        return u_poly_study2(alpha, area, theta, center)
    return C.shape_polygon(shape, alpha, s, theta, center=center, area=area)


def eyes_from_poly(poly0, size, theta, center):
    """Two dark disks + highlights in the upper interior of an unrotated polygon (apex-down frame)."""
    c = np.asarray(center)
    top, bot = poly0[:, 1].min(), poly0[:, 1].max()
    ye = top + 0.30 * (bot - top)
    m = C.fill_mask(poly0)
    row = m[int(np.clip(round(ye), 0, 223))]
    xs = np.where(row > 0.5)[0]
    if len(xs) < 4:
        return np.zeros((224, 224), np.float32), np.zeros((224, 224), np.float32)
    xl, xr = xs.min(), xs.max(); width = xr - xl; xm = (xl + xr) / 2
    sep, rad = 0.45 * width, min(0.07 * size, 0.14 * width)
    SS = C.SS
    imE = Image.new("L", (224 * SS, 224 * SS), 0); imH = Image.new("L", (224 * SS, 224 * SS), 0)
    from PIL import ImageDraw
    dE, dH = ImageDraw.Draw(imE), ImageDraw.Draw(imH)
    for sx in (-1, 1):
        e = C._rot(np.array([[xm + sx * sep / 2, ye]]), theta, c)[0] * SS
        R = rad * SS
        dE.ellipse([e[0] - R, e[1] - R, e[0] + R, e[1] + R], fill=255)
        h = e - 0.35 * R; rh = 0.3 * R
        dH.ellipse([h[0] - rh, h[1] - rh, h[0] + rh, h[1] + rh], fill=255)
    f = lambda im: np.asarray(im.resize((224, 224), Image.LANCZOS), np.float32) / 255
    return f(imE), f(imH)

# ------------------------------------------------------------------ content
def load_crop(path):
    img, _ = C.center_crop_224(C.load_rgb(path))
    return u8(C.to01(img))


def get_scan(probe):
    f = os.path.join(C.WORK, f"scan4d_unit{args.unit}.npz")
    if not os.path.exists(f):
        s0 = np.load(os.path.join(C.WORK, "scan_imnv2.npz")); p0 = np.load(os.path.join(C.WORK, "scan_pets.npz"))
        def scan(paths):
            outs = {}
            with ThreadPoolExecutor(8) as ex:
                for s in range(0, len(paths), 64):
                    o = U.run_arrays(probe, list(ex.map(load_crop, paths[s:s + 64])))
                    for k, v in o.items():
                        outs.setdefault(k, []).append(v)
            return {k: np.concatenate(v) for k, v in outs.items()}
        ri, rp = scan(list(s0["paths"])), scan(list(p0["paths"]))
        out = {"imn_" + k: v for k, v in ri.items()}
        out.update({"pets_" + k: v for k, v in rp.items()})
        out.update(imn_cls=s0["cls"], imn_paths=s0["paths"], pets_paths=p0["paths"], pets_breed=p0["breed"],
                   pets_species=p0["species"])
        np.savez(f, **out)
    return dict(np.load(f))


def fur_patches(pets, rng, n):
    out = []
    cand = [d for d in pets if d["head"] is not None]
    rng.shuffle(cand)
    for d in cand:
        if len(out) >= n:
            break
        img = C.load_rgb(d["path"]); fg = np.asarray(Image.open(d["trimap"])) == 1
        x0, y0, x1, y1 = d["head"]; hw, hh = x1 - x0, y1 - y0
        bx0, by0, bx1, by1 = x0 - 0.1 * hw, y0 - 0.1 * hh, x1 + 0.1 * hw, y1 + 0.1 * hh
        H, W = fg.shape
        done = False
        for P in (160, 128, 96, 72):
            if P >= min(H, W):
                continue
            for _ in range(30):
                l, t = rng.integers(0, W - P), rng.integers(0, H - P)
                if l < bx1 and l + P > bx0 and t < by1 and t + P > by0:
                    continue
                if fg[t:t + P, l:l + P].mean() >= 0.95:
                    out.append(C.to01(img.crop((l, t, l + P, t + P)).resize((224, 224), Image.BICUBIC)))
                    done = True; break
            if done:
                break
    return out


def head_canvas(d, scale_to=None, box_frac=None):
    img = C.load_rgb(d["path"])
    x0, y0, x1, y1 = d["head"]
    hs = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = (scale_to / hs) if scale_to else (224 / (box_frac * hs))
    w, h = img.size
    im = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BICUBIC)
    can = Image.new("RGB", (224, 224), tuple(int(v * 255) for v in C.GRAY))
    can.paste(im, (round(112 - cx * k), round(112 - cy * k)))
    return C.to01(can)


def translate_to_center(img01, iy, ix):
    py, px = 16 * iy + 8, 16 * ix + 8
    can = C.gray_canvas(); dy, dx = 112 - py, 112 - px
    ys, xs = slice(max(0, dy), min(224, 224 + dy)), slice(max(0, dx), min(224, 224 + dx))
    yo, xo = slice(max(0, -dy), min(224, 224 - dy)), slice(max(0, -dx), min(224, 224 - dx))
    can[ys, xs] = img01[yo, xo]
    return can

# ------------------------------------------------------------------ routes
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def route(cpos):
    m = cpos.mean(0); order = np.argsort(-m)
    k = int(np.searchsorted(np.cumsum(m[order]) / m.sum(), 0.5) + 1)
    return [int(c) for c in order[:k]]


def share(cpos, rt):
    return float(cpos[:, rt].sum(1).mean() / cpos.sum(1).mean())


def route_analysis(groups_raw, baseline, rs):
    G = {}
    for g, (cv, rl) in groups_raw.items():
        thr = 0.2; keep = rl >= thr
        if keep.sum() < 30:
            thr = 0.1; keep = rl >= thr
        cpos = np.clip(cv[keep] - (baseline if baseline is not None else 0), 0, None)
        perm = rs.permutation(len(cpos))
        G[g] = {"n": int(len(cpos)), "threshold": thr, "underpowered": bool(len(cpos) < 30),
                "A": cpos[perm[: len(perm) // 2]], "B": cpos[perm[len(perm) // 2:]]}
    names = []
    for g, d in G.items():
        if len(d["A"]) >= 2 and len(d["B"]) >= 2 and d["A"].sum() > 0 and d["B"].sum() > 0:
            d["route"] = route(d["A"]); d["split_half_J"] = jaccard(d["route"], route(d["B"]))
            names.append(g)
    pairs = {}
    for h in names:
        own = share(G[h]["B"], G[h]["route"])
        for g in names:
            if g != h:
                pairs[f"X_{h}<-{g}"] = share(G[h]["B"], G[g]["route"]) / own if own > 0 else np.nan
                pairs[f"J_{h},{g}"] = jaccard(G[h]["route"], G[g]["route"])
    return {"groups": {g: {k: v for k, v in d.items() if k not in ("A", "B")} for g, d in G.items()},
            "pairs": pairs}

# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(22)
    probe = U.UnitProbe(args.unit)
    cats = C.imagenet_categories()
    aref4d = np.load(os.path.join(C.WORK, "aref.npz"))["aref4d"]
    A = float(aref4d[args.unit])
    st1 = json.load(open(os.path.join(OUT, "stage1_tuning.json")))
    TH, AL, SS_ = st1["theta_star"], st1["alpha_star"], st1["s_star"]
    log(f"unit 4d:{args.unit}  A_ref {A:.1f}  theta* {TH} alpha* {AL} s* {SS_}")
    S = get_scan(probe)
    imn_rel, imn_cls, pets_rel = S["imn_drive"] / A, S["imn_cls"], S["pets_drive"] / A
    pets = C.pets_index()
    w = probe.w
    g_cats = C.resolve_names(C.CATS, cats, LOG); g_wild = C.resolve_names(C.WILD_CANIDS, cats, LOG)
    g_veh = C.resolve_names(C.VEHICLES, cats, LOG)

    dtd = [p for c in C.DTD_STAGE2 for p in C.dtd_paths([c])[c]]
    fur = fur_patches(pets, rng, 400); furA, furB = fur[:200], fur[200:]
    log(f"fur patches {len(fur)}")

    # ============================================ stage 2 synthetic (P1, P2, P5)
    area_s = C.polygon_area("V", AL, SS_)
    stims, meta = [], []
    for shape in ["V", "U", "E", "O"]:
        for fill in ["neutral", "fur", "fur_scr"]:
            for eyes in [0, 1]:
                for _ in range(60):
                    th = TH + rng.choice([-15, 0, 15]); s = float(rng.choice([SS_, 0.8 * SS_]))
                    ar_ = area_s * (s / SS_) ** 2
                    ctr = (112 + rng.uniform(-8, 8), 112 + rng.uniform(-8, 8))
                    m = C.fill_mask(poly_for(shape, AL, s, th, ar_, ctr))
                    if fill == "neutral":
                        tex = C.random_texture(dtd[rng.integers(len(dtd))], rng)
                    else:
                        tex = furA[rng.integers(len(furA))]
                        if fill == "fur_scr":
                            tex = C.phase_scramble(tex, rng)
                    img = C.composite(C.gray_canvas(), tex, m)
                    if eyes:
                        E, Hh = eyes_from_poly(poly_for(shape, AL, s, 0, ar_, ctr), s, th, ctr)
                        img = C.composite(img, np.full_like(img, 0.06), E)
                        img = C.composite(img, np.full_like(img, 0.97), Hh)
                    stims.append(u8(img)); meta.append((shape, fill, eyes))
    r2 = U.run_arrays(probe, stims, keys={"drive", "max_pop", "route_at"})
    rel2 = r2["drive"] / A; pop2 = r2["max_pop"] / aref4d[None, :]
    M = np.array(meta, dtype=object)
    def sel(shape=None, fill=None, eyes=None):
        k = np.ones(len(M), bool)
        for j, v in enumerate([shape, fill, eyes]):
            if v is not None:
                k &= np.array([x == v for x in M[:, j]])
        return k
    cells = {f"{s}|{f}|eyes{e}": mean_ci(rel2[sel(s, f, e)]) for s in "VUEO" for f in ["neutral", "fur", "fur_scr"] for e in (0, 1)}
    v, u, e_, o = (rel2[sel(s, "neutral", 0)] for s in "VUEO")
    P1 = {"V": mean_ci(v), "U": mean_ci(u), "E": mean_ci(e_), "O": mean_ci(o),
          "V_minus_E": diff_ci(v, e_), "V_minus_U": diff_ci(v, u), "U_minus_E": diff_ci(u, e_),
          "E_minus_O": diff_ci(e_, o), "V_minus_O": diff_ci(v, o)}
    P1["HP_pass"] = bool(P1["V"]["mean"] >= 0.30 and big(P1["V_minus_E"]) and big(P1["V_minus_U"]))
    P1["HE_pattern"] = bool(approx(P1["V_minus_U"]) and approx(P1["V_minus_E"]) and approx(P1["U_minus_E"])
                            and big(P1["E_minus_O"]))
    P1["HC_all_below_0.30"] = bool(max(P1[s]["mean"] for s in "VUEO") < 0.30)
    kV, kE, kO, kU = (sel(s, "neutral", 0) for s in "VEOU")
    dVE = pop2[kV].mean(0) - pop2[kE].mean(0); dVO = pop2[kV].mean(0) - pop2[kO].mean(0)
    dVU = pop2[kV].mean(0) - pop2[kU].mean(0)
    rk = lambda d: int((d > d[args.unit]).sum() + 1)
    P2 = {"delta_VE_unit": float(dVE[args.unit]), "rank_VE_of_528": rk(dVE), "rank_VO_of_528": rk(dVO),
          "rank_VU_of_528": rk(dVU), "delta_VE_pct": {q: float(np.percentile(dVE, q)) for q in (50, 90, 95, 99)},
          "HP_pass": bool(rk(dVE) <= 27)}
    P5 = {"O_eyes_minus_O": diff_ci(rel2[sel("O", "neutral", 1)], o),
          "V_eyes_minus_V": diff_ci(rel2[sel("V", "neutral", 1)], v),
          "E_eyes_minus_E": diff_ci(rel2[sel("E", "neutral", 1)], e_)}
    furfx = {f"{s}_fur_minus_neutral": diff_ci(rel2[sel(s, "fur", 0)], rel2[sel(s, "neutral", 0)]) for s in "VUEO"}
    log(f"P1 V {P1['V']['mean']:.3f} U {P1['U']['mean']:.3f} E {P1['E']['mean']:.3f} O {P1['O']['mean']:.3f}  P2 rank VE {P2['rank_VE_of_528']}  ({time.time() - t0:.0f}s)")

    # ============================================ P3 face-sized apertures
    al_ap, h_face = AL, None
    for al_try in [AL, 90]:
        for h in [160, 144, 128, 112, 96, 80, 64]:
            ar_ = np.pi * (h / 2) ** 2
            polys = [poly_for("V", al_try, h, TH, ar_), poly_for("U", al_try, h, TH, ar_),
                     poly_for("E", al_try, h, TH, ar_), poly_for("O", al_try, h, TH, ar_)]
            if all(np.ptp(p[:, 0]) <= 216 and np.ptp(p[:, 1]) <= 216 for p in polys):
                h_face, al_ap = h, al_try; break
        if h_face:
            break
    if al_ap != AL:
        log(f"P3: no h fits at alpha*={AL}; apertures use alpha=90")
    area_f = np.pi * (h_face / 2) ** 2
    masks = {s: C.fill_mask(poly_for(s, al_ap, h_face, TH, area_f)) for s in "VUEO"}
    log(f"P3 face size h={h_face}px, aperture alpha={al_ap}, area={area_f:.0f}px2")
    with_head = [i for i, d in enumerate(pets) if d["head"] is not None and
                 max(d["head"][2] - d["head"][0], d["head"][3] - d["head"][1]) >= 40]
    cat_ids = [i for i in with_head if pets[i]["species"] == "cat"]; dog_ids = [i for i in with_head if pets[i]["species"] == "dog"]
    rng.shuffle(cat_ids); rng.shuffle(dog_ids)
    order = np.argsort(-imn_rel)
    top_mam = [i for i in order[:100] if int(imn_cls[i]) in MAMMALS]
    def tr(i):
        iy, ix = S["imn_argmax_yx"][i]
        return translate_to_center(C.to01(C.center_crop_224(C.load_rgb(S["imn_paths"][i]))[0]), iy, ix)
    contents = {"cat_faces": [head_canvas(pets[i], scale_to=h_face) for i in cat_ids[:300]],
                "dog_faces": [head_canvas(pets[i], scale_to=h_face) for i in dog_ids[:300]],
                "top_mammal": [tr(i) for i in top_mam],
                "neutral": [C.random_texture(dtd[rng.integers(len(dtd))], rng) for _ in range(200)]}
    f2, fm = [], []
    for k, lst in contents.items():
        for j, img in enumerate(lst):
            f2.append(u8(img)); fm.append((k, j, "full"))
            for s, m in masks.items():
                f2.append(u8(C.composite(C.gray_canvas(), img, m))); fm.append((k, j, s))
    relf = U.run_arrays(probe, f2, keys={"drive"})["drive"] / A
    fm_arr = np.array(fm, dtype=object)
    tabs = {k: {a: relf[(fm_arr[:, 0] == k) & (fm_arr[:, 2] == a)] for a in ["full", "V", "U", "E", "O"]} for k in contents}
    P3 = {"h_face": h_face, "aperture_alpha": al_ap}
    for k in contents:
        t = tabs[k]
        P3[k] = {a: mean_ci(t[a]) for a in t}
        for a, b in [("V", "O"), ("V", "E"), ("V", "U"), ("E", "O")]:
            P3[k][f"{a}_minus_{b}"] = diff_ci(t[a], t[b], paired=True)
    def sigma(kv, ko, nv, no):
        Sh = 0.5 * ((kv - ko) + (nv - no)); Cc = 0.5 * ((kv - nv) + (ko - no))
        Sp, Cp = max(Sh, 0), max(Cc, 0)
        return (np.nan if Sp + Cp <= 0 else float(Sp / (Sp + Cp))), Sh, Cc
    rr = np.random.default_rng(5)
    for k in ["cat_faces", "dog_faces", "top_mammal"]:
        kv, ko, nv, no = tabs[k]["V"], tabs[k]["O"], tabs["neutral"]["V"], tabs["neutral"]["O"]
        s0_, Sh, Cc = sigma(kv.mean(), ko.mean(), nv.mean(), no.mean())
        bs = []
        for _ in range(4000):
            i1 = rr.integers(0, len(kv), len(kv)); i2 = rr.integers(0, len(nv), len(nv))
            bs.append(sigma(kv[i1].mean(), ko[i1].mean(), nv[i2].mean(), no[i2].mean())[0])
        P3[k]["sigma"] = {"sigma": s0_, "S": Sh, "C": Cc, "lo": float(np.nanquantile(bs, .025)), "hi": float(np.nanquantile(bs, .975))}
    best_neutral = max(tabs["neutral"][a].mean() for a in "VUEO")
    for k in ["cat_faces", "dog_faces"]:
        P3[k]["HP_pass"] = bool(big(P3[k]["V_minus_O"]) and big(P3[k]["V_minus_E"]))
        P3[k]["HE_pattern"] = bool(approx(P3[k]["V_minus_E"]) and big(P3[k]["E_minus_O"]))
        P3[k]["HC_sigma_le_0.33"] = bool(P3[k]["sigma"]["sigma"] <= 0.33)
        P3[k]["full_ge_2x_best_neutral_aperture"] = bool(tabs[k]["full"].mean() >= 2 * best_neutral)
    log(f"P3 cat V-O {P3['cat_faces']['V_minus_O']['diff']:.3f} V-E {P3['cat_faces']['V_minus_E']['diff']:.3f} sigma {P3['cat_faces']['sigma']['sigma']}  ({time.time() - t0:.0f}s)")

    # ============================================ P4 routes (512 pooled mixed4c inputs)
    imn_c, pets_c = S["imn_route_at"] * w, S["pets_route_at"] * w
    psp = S["pets_species"]
    groups_raw = {
        "cats": (np.concatenate([imn_c[np.isin(imn_cls, g_cats)], pets_c[psp == "cat"]]),
                 np.concatenate([imn_rel[np.isin(imn_cls, g_cats)], pets_rel[psp == "cat"]])),
        "dogs": (np.concatenate([imn_c[np.isin(imn_cls, C.DOG_IDX)], pets_c[psp == "dog"]]),
                 np.concatenate([imn_rel[np.isin(imn_cls, C.DOG_IDX)], pets_rel[psp == "dog"]])),
        "wild_canids": (imn_c[np.isin(imn_cls, g_wild)], imn_rel[np.isin(imn_cls, g_wild)]),
        "vehicles": (imn_c[np.isin(imn_cls, g_veh)], imn_rel[np.isin(imn_cls, g_veh)]),
        "V_silhouettes": (r2["route_at"][kV] * w, rel2[kV]),
    }
    rs = np.random.default_rng(3)
    P4 = route_analysis(groups_raw, None, rs)
    base = imn_c[rs.choice(len(imn_c), 500, replace=False)].mean(0)
    P4b = route_analysis(groups_raw, base, rs)
    pr = P4["pairs"]
    vg = P4["groups"]["V_silhouettes"]
    P4["V_powered"] = bool(not vg["underpowered"] and "route" in vg)
    P4["X_V<-cat"] = pr.get("X_V_silhouettes<-cats", np.nan); P4["X_cat<-V"] = pr.get("X_cats<-V_silhouettes", np.nan)
    P4["HP_shared"] = bool(P4["V_powered"] and P4["X_V<-cat"] >= 0.5 and P4["X_cat<-V"] >= 0.5)
    P4["HC_not_shared"] = bool(P4["V_powered"] and P4["X_V<-cat"] <= 0.25)
    maxdesc = S["imn_max_desc"]
    def describe(ch):
        top = np.argsort(-maxdesc[:, ch])[:30]
        cl, cnt = np.unique(imn_cls[top], return_counts=True)
        return [cats[int(cl[j])] for j in np.argsort(-cnt)[:3]]
    P4["route_channel_descriptions_mixed4c"] = {g: {int(ch): describe(ch) for ch in d.get("route", [])[:5]}
                                                for g, d in P4["groups"].items()}
    log(f"P4 X_V<-cat {P4['X_V<-cat']} X_cat<-V {P4['X_cat<-V']} powered {P4['V_powered']}  ({time.time() - t0:.0f}s)")

    # ============================================ P6 silhouettes
    P6 = silhouettes(probe, S, imn_rel, pets_rel, pets, A, rng)
    sf, ss = P6["all"]["sil_flat"]["R_mean"]["mean"], P6["all"]["sil_scr"]["R_mean"]["mean"]
    P6["HP_pass"] = bool(sf >= 0.5 and ss >= 0.5); P6["HC_pass"] = bool(sf <= 0.25 and ss <= 0.25)
    log(f"P6 R sil_flat {sf:.3f} sil_scr {ss:.3f}  ({time.time() - t0:.0f}s)")

    # ============================================ secondary F6, F7
    def pets_heads(breeds):
        ids = [i for i in with_head if pets[i]["breed"] in breeds]
        imgs = [u8(head_canvas(pets[i], box_frac=1.5)) for i in ids]
        return U.run_arrays(probe, imgs, keys={"drive"})["drive"] / A if imgs else np.array([])
    F6 = {}
    for name, pt, rd in [("pets_cats", C.F6_PETS_CAT_POINTED, C.F6_PETS_CAT_ROUND),
                         ("pets_dogs", C.F6_PETS_DOG_POINTED, C.F6_PETS_DOG_ROUND)]:
        a, b = pets_heads(pt), pets_heads(rd)
        F6[name] = {"pointed": mean_ci(a), "round": mean_ci(b), "pointed_minus_round": diff_ci(a, b)}
        for br in pt + rd:
            F6[name][f"breed:{br}"] = mean_ci(pets_heads([br]))
    ip, ir = C.resolve_names(C.F6_IMNV2_POINTED, cats, []), C.resolve_names(C.F6_IMNV2_ROUND, cats, [])
    F6["imagenetv2"] = {"pointed": mean_ci(imn_rel[np.isin(imn_cls, ip)]), "round": mean_ci(imn_rel[np.isin(imn_cls, ir)]),
                        "pointed_minus_round": diff_ci(imn_rel[np.isin(imn_cls, ip)], imn_rel[np.isin(imn_cls, ir)]),
                        "Siamese_minus_Persian": diff_ci(imn_rel[imn_cls == cats.index("Siamese cat")],
                                                         imn_rel[imn_cls == cats.index("Persian cat")])}
    cls_mean = np.array([imn_rel[imn_cls == c].mean() for c in range(1000)])
    cls_pre = np.array([S["imn_pre_mean"][imn_cls == c].mean() for c in range(1000)])
    F7 = {"all_class_median_rel": float(np.median(cls_mean)), "all_image_median_pre_mean": float(np.median(S["imn_pre_mean"])),
          "top20_classes": [(cats[c], float(cls_mean[c])) for c in np.argsort(-cls_mean)[:20]]}
    for name, lst in [("pointed", C.F7_POINTED), ("round", C.F7_ROUND), ("essay_examples", C.F7_ESSAY)]:
        idx = C.resolve_names(lst, cats, [])
        F7[name] = {"mean_rel": float(cls_mean[idx].mean()), "mean_pre_mean": float(cls_pre[idx].mean()),
                    "classes": {cats[c]: {"rel": float(cls_mean[c]), "rank": int((cls_mean > cls_mean[c]).sum() + 1)} for c in idx}}
    F7["pointed_minus_round"] = C.boot_diff(cls_mean[C.resolve_names(C.F7_POINTED, cats, [])],
                                            cls_mean[C.resolve_names(C.F7_ROUND, cats, [])])

    # ============================================ decisions (as pre-registered)
    v1 = json.load(open(os.path.join(OUT, "stage1_tuning.json")))["V1_pass"]
    both = lambda key: all(P3[k][key] for k in ["cat_faces", "dog_faces"])
    p3_weak_shape = all((P3[k]["V_minus_O"]["diff"] < 0.10) or (P3[k]["V_minus_E"]["diff"] < 0.10) for k in ["cat_faces", "dog_faces"])
    dec = {
        "V1_pass": v1, "P1_HP_pass": P1["HP_pass"], "P1_HE_pattern": P1["HE_pattern"], "P1_HC_all_below_0.30": P1["HC_all_below_0.30"],
        "P2_HP_pass": P2["HP_pass"], "P3_HP_pass_both": both("HP_pass"), "P3_HE_pattern_both": both("HE_pattern"),
        "P3_HC_sigma_both": both("HC_sigma_le_0.33"), "P3_HC_full_ge_2x_both": both("full_ge_2x_best_neutral_aperture"),
        "P4_V_powered": P4["V_powered"], "P4_HP_shared": P4["HP_shared"], "P4_HC_not_shared": P4["HC_not_shared"],
        "P5_eye_effect": P5["O_eyes_minus_O"]["diff"], "P6_HP_pass": P6["HP_pass"], "P6_HC_pass": P6["HC_pass"],
    }
    dec["H_P_supported"] = bool(P1["HP_pass"] and P2["HP_pass"] and both("HP_pass") and P6["HP_pass"]
                                and (P4["HP_shared"] or not P4["V_powered"]))
    dec["H_P_refuted"] = bool(v1 and not P1["HP_pass"] and p3_weak_shape and P6["HC_pass"])
    dec["H_C_supported"] = bool(P1["HC_all_below_0.30"] and both("HC_sigma_le_0.33") and P6["HC_pass"]
                                and (P4["HC_not_shared"] or not P4["V_powered"]))
    dec["H_E_supported"] = bool(P1["HE_pattern"] and both("HE_pattern"))
    dec["H_face_P5"] = bool(P5["O_eyes_minus_O"]["diff"] >= 0.15)
    out = {"unit": f"4d:{args.unit}", "A_ref": A, "theta_star": TH, "alpha_star": AL, "s_star": SS_,
           "P1": P1, "P2": P2, "P3": P3, "P4": P4, "P4_baseline_subtracted": P4b, "P5": P5, "P6": P6,
           "fur_effects": furfx, "stage2_cells": cells, "F6": F6, "F7": F7, "decisions": dec, "log": LOG,
           "runtime_s": time.time() - t0}
    save(out, f"stage2_tests{args.tag}.json")
    log(json.dumps(dec, indent=1, default=str))
    # example sheet: stage-2 shapes and apertures
    sheet = Image.new("RGB", (8 * 112, 3 * 112), (255, 255, 255))
    ex = [sel(s, "neutral", 0) for s in "VUEO"] + [sel(s, "fur", 1) for s in "VUEO"]
    for j, k in enumerate(ex):
        sheet.paste(Image.fromarray(stims[int(np.where(k)[0][0])]).resize((112, 112)), (j * 112, 0))
    for row, kname in [(1, "cat_faces"), (2, "dog_faces")]:
        ids = [i for i, m in enumerate(fm) if m[0] == kname and m[1] in (0,)] + \
              [i for i, m in enumerate(fm) if m[0] == kname and m[1] in (1,)]
        for j, i in enumerate(ids[:8]):
            sheet.paste(Image.fromarray(f2[i]).resize((112, 112)), (j * 112, row * 112))
    sheet.save(os.path.join(OUT, f"fig_stage2_examples{args.tag}.jpg"), quality=88)


def silhouettes(probe, S, imn_rel, pets_rel, pets, A, rng):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    det = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    items, excl = [], 0
    for i in np.argsort(-imn_rel)[:100]:
        img = C.to01(C.center_crop_224(C.load_rgb(S["imn_paths"][i]))[0])
        iy, ix = S["imn_argmax_yx"][i]; py, px = 16 * iy + 8, 16 * ix + 8
        with torch.no_grad():
            o = det([torch.from_numpy(img).permute(2, 0, 1)])[0]
        best = None
        for k in range(len(o["scores"])):
            if o["scores"][k] < 0.5:
                continue
            mk = o["masks"][k, 0].numpy() > 0.5
            if mk[py, px] and (best is None or o["scores"][k] > o["scores"][best[0]]):
                best = (k, mk)
        if best is None:
            excl += 1; continue
        items.append(("imagenetv2", img, best[1].astype(np.float32)))
    for i in np.argsort(-pets_rel)[:100]:
        d = pets[i]
        img_pil, (s, l, t) = C.center_crop_224(C.load_rgb(d["path"]))
        tri = Image.open(d["trimap"]); W, H = tri.size
        tri = np.asarray(tri.resize((max(224, round(W * s)), max(224, round(H * s))), Image.NEAREST).crop((l, t, l + 224, t + 224)))
        items.append(("pets", C.to01(img_pil), ((tri == 1) | (tri == 3)).astype(np.float32)))
    imgs, meta = [], []
    for j, (src, img, m) in enumerate(items):
        mo = img[m > 0.5].mean(0) if (m > 0.5).any() else C.GRAY
        mb = img[m < 0.5].mean(0) if (m < 0.5).any() else C.GRAY
        obj_only = img * m[..., None] + mo * (1 - m[..., None]); bg_only = img * (1 - m[..., None]) + mb * m[..., None]
        edits = {"orig": img, "sil_flat": C.composite(img, np.broadcast_to(mo, img.shape), m),
                 "sil_scr": C.composite(img, C.phase_scramble(obj_only, rng), m),
                 "bg_flat": C.composite(np.broadcast_to(mb, img.shape).copy(), img, m),
                 "bg_scr": C.composite(C.phase_scramble(bg_only, rng), img, m)}
        for k, v in edits.items():
            imgs.append(u8(v)); meta.append((src, j, k))
    rel = U.run_arrays(probe, imgs, keys={"drive"})["drive"] / A
    meta = np.array(meta, dtype=object)
    res = {"n_imagenetv2": sum(x[0] == "imagenetv2" for x in items), "excluded_no_mask": excl,
           "n_pets": sum(x[0] == "pets" for x in items)}
    for src in ["imagenetv2", "pets", "all"]:
        keep = np.ones(len(meta), bool) if src == "all" else (meta[:, 0] == src)
        orig = rel[keep & (meta[:, 2] == "orig")]
        res[src] = {"orig_rel": mean_ci(orig)}
        for k in ["sil_flat", "sil_scr", "bg_flat", "bg_scr"]:
            ed = rel[keep & (meta[:, 2] == k)]
            R = ed / np.maximum(orig, 1e-6)
            res[src][k] = {"R_mean": mean_ci(R), "R_median": float(np.median(R)), "rel_edit": mean_ci(ed)}
    return res


if __name__ == "__main__":
    main()
