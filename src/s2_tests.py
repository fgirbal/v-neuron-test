"""Stage 2: pre-registered tests P1-P5 and secondary tests F3, F6, F7 for one mixed4e 1x1 unit.

Usage: python -I s2_tests.py --unit 55            (the reported run)
       python -I s2_tests.py --unit 54 --tag _debug54   (pipeline debugging on a different unit)
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--unit", type=int, default=55)
ap.add_argument("--tag", default="")
args = ap.parse_args()
C.UNIT = args.unit
LOG = []


def u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

# ============================================================== scans for this unit
def load_crop(path):
    img, _ = C.center_crop_224(C.load_rgb(path))
    return u8(C.to01(img))


def scan_paths(probe, paths):
    outs = {}
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(paths), 64):
            arrs = list(ex.map(load_crop, paths[s:s + 64]))
            o = C.run_arrays(probe, arrs, keys={"drive", "pre_mean", "argmax_yx", "a4d_at"})
            for k, v in o.items():
                outs.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in outs.items()}


def unit_scans(probe):
    f = os.path.join(C.WORK, f"scan_unit{C.UNIT}.npz")
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=False))
    s0 = np.load(os.path.join(C.WORK, "scan_imnv2.npz"))
    p0 = np.load(os.path.join(C.WORK, "scan_pets.npz"))
    ri = scan_paths(probe, list(s0["paths"]))
    rp = scan_paths(probe, list(p0["paths"]))
    out = {"imn_" + k: v for k, v in ri.items()}
    out.update({"pets_" + k: v for k, v in rp.items()})
    out.update(imn_cls=s0["cls"], imn_paths=s0["paths"], imn_max4d=s0["max4d"],
               pets_paths=p0["paths"], pets_breed=p0["breed"], pets_species=p0["species"])
    np.savez(f, **out)
    return out

# ============================================================== content pools
def fur_patches(pets, rng, n):
    """Square patches fully inside the animal mask and away from the head box."""
    out = []
    cand = [d for d in pets if d["head"] is not None]
    rng.shuffle(cand)
    for d in cand:
        if len(out) >= n:
            break
        img = C.load_rgb(d["path"]); tri = np.asarray(Image.open(d["trimap"]))
        fg = (tri == 1)
        x0, y0, x1, y1 = d["head"]
        hw, hh = x1 - x0, y1 - y0
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
                    done = True
                    break
            if done:
                break
    return out


def head_canvas(d, target, scale_to=None, box_frac=None):
    """Pets image placed on a gray canvas with the head centre at the canvas centre.
    scale_to: head size in px after scaling; box_frac: crop a square of box_frac*head and resize to 224."""
    img = C.load_rgb(d["path"])
    x0, y0, x1, y1 = d["head"]
    hs = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = (scale_to / hs) if scale_to else (224 / (box_frac * hs))
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

# ============================================================== helpers
def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    m, lo, hi = C.boot_ci(x)
    return {"n": int(len(x)), "mean": m, "lo": lo, "hi": hi}


def diff_ci(a, b, paired=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        m, lo, hi = C.boot_ci(a - b)
    else:
        m, lo, hi = C.boot_diff(a, b)
    return {"diff": m, "lo": lo, "hi": hi, "n_a": int(len(a)), "n_b": int(len(b))}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def route(cpos):
    m = cpos.mean(0)
    order = np.argsort(-m)
    cum = np.cumsum(m[order]) / m.sum()
    k = int(np.searchsorted(cum, 0.5) + 1)
    return [int(c) for c in order[:k]]


def share(cpos, rt):
    return float(cpos[:, rt].sum(1).mean() / cpos.sum(1).mean())

# ============================================================== main
def main():
    t0 = time.time()
    rng = np.random.default_rng(2)
    probe = C.Probe()
    cats = C.imagenet_categories()
    ar = np.load(os.path.join(C.WORK, "aref.npz"))
    aref4e, aref4d = ar["aref4e"], ar["aref4d"]
    A = float(aref4e[C.UNIT])
    st1 = json.load(open(os.path.join(C.RESULTS, "stage1_tuning.json")))
    TH, AL, SS_ = st1["theta_star"], st1["alpha_star"], st1["s_star"]
    log(f"unit {C.UNIT}  A_ref {A:.1f}  theta* {TH} alpha* {AL} s* {SS_}")

    S = unit_scans(probe)
    imn_rel, imn_cls = S["imn_drive"] / A, S["imn_cls"]
    pets_rel = S["pets_drive"] / A
    pets = C.pets_index()
    assert len(pets) == len(pets_rel)
    w = probe.w55  # weights of the probed unit (Probe reads C.UNIT)

    names_log = []
    g_cats = C.resolve_names(C.CATS, cats, names_log)
    g_wild = C.resolve_names(C.WILD_CANIDS, cats, names_log)
    g_veh = C.resolve_names(C.VEHICLES, cats, names_log)
    for m_ in names_log:
        log(m_)

    # ---------------- pools
    dtd2 = C.dtd_paths(C.DTD_STAGE2)
    dtd2_all = [p for c in C.DTD_STAGE2 for p in dtd2[c]]
    fur = fur_patches(pets, rng, 400)
    log(f"fur patches: {len(fur)}")
    furA, furB = fur[: len(fur) // 2], fur[len(fur) // 2:]
    scene_idx = [i for c in C.resolve_names(C.SCENE_CLASSES, cats, []) for i in np.where(imn_cls == c)[0]]
    scene_idx = [i for i in scene_idx if imn_rel[i] < 0.05][:20]
    scenes = [C.to01(C.center_crop_224(C.load_rgb(S["imn_paths"][i]))[0]) for i in scene_idx]
    log(f"scene backgrounds: {len(scenes)}")

    # ============================================================ F1 stage 2
    area = C.polygon_area("V", AL, SS_)
    cells, stims, meta = [], [], []
    for shape in ["V", "U", "O"]:
        for fill in ["neutral", "fur", "fur_scr"]:
            for eyes in [0, 1]:
                for bg in ["gray", "scene"]:
                    for _ in range(60):
                        th = TH + rng.choice([-15, 0, 15])
                        s = float(rng.choice([SS_, 0.8 * SS_]))
                        ar_ = area * (s / SS_) ** 2
                        ctr = (112 + rng.uniform(-8, 8), 112 + rng.uniform(-8, 8))
                        poly = C.shape_polygon(shape, AL, s, th, center=ctr, area=ar_)
                        m = C.fill_mask(poly)
                        if fill == "neutral":
                            tex = C.random_texture(dtd2_all[rng.integers(len(dtd2_all))], rng)
                        else:
                            tex = furA[rng.integers(len(furA))]
                            if fill == "fur_scr":
                                tex = C.phase_scramble(tex, rng)
                        base = C.gray_canvas() if bg == "gray" else scenes[rng.integers(len(scenes))]
                        img = C.composite(base, tex, m)
                        if eyes:
                            E, Hh = C.eyes_layer(shape, AL, s, th, ar_, center=ctr)
                            img = C.composite(img, np.full_like(img, 0.06), E)
                            img = C.composite(img, np.full_like(img, 0.97), Hh)
                        stims.append(u8(img)); meta.append((shape, fill, eyes, bg))
    log(f"stage-2 stimuli: {len(stims)}  ({time.time() - t0:.0f}s)")
    r2 = C.run_arrays(probe, stims, keys={"drive", "max4e", "a4d_at"})
    rel2 = r2["drive"] / A
    pop2 = r2["max4e"] / aref4e[None, :]
    M = np.array(meta, dtype=object)
    def sel(shape=None, fill=None, eyes=None, bg=None):
        k = np.ones(len(M), bool)
        for j, v in enumerate([shape, fill, eyes, bg]):
            if v is not None:
                k &= np.array([x == v for x in M[:, j]])
        return k
    cell_table = {}
    for shape in ["V", "U", "O"]:
        for fill in ["neutral", "fur", "fur_scr"]:
            for eyes in [0, 1]:
                for bg in ["gray", "scene"]:
                    cell_table[f"{shape}|{fill}|eyes{eyes}|{bg}"] = mean_ci(rel2[sel(shape, fill, eyes, bg)])
    vN = rel2[sel("V", "neutral", 0, "gray")]; uN = rel2[sel("U", "neutral", 0, "gray")]
    oN = rel2[sel("O", "neutral", 0, "gray")]; oE = rel2[sel("O", "neutral", 1, "gray")]
    P1 = {"V": mean_ci(vN), "U": mean_ci(uN), "O": mean_ci(oN),
          "V_minus_O": diff_ci(vN, oN), "V_minus_U": diff_ci(vN, uN)}
    P1["HV_pass"] = bool(P1["V"]["mean"] >= 0.30 and P1["V_minus_O"]["diff"] >= 0.20 and P1["V_minus_O"]["lo"] > 0
                         and P1["V_minus_U"]["diff"] >= 0.10 and P1["V_minus_U"]["lo"] > 0)
    _, lo90, hi90 = C.boot_diff(vN, oN, alpha=0.10)  # equivalence uses a 90% CI (pre-registered)
    P1["V_minus_O_90ci"] = [lo90, hi90]
    P1["null_pattern_pass"] = bool(P1["V"]["mean"] < 0.30 and P1["O"]["mean"] < 0.30 and
                                   lo90 > -0.10 and hi90 < 0.10)
    P5 = {"O_eyes_minus_O": diff_ci(oE, oN), "scene_bg": diff_ci(rel2[sel("O", "neutral", 1, "scene")],
                                                                  rel2[sel("O", "neutral", 0, "scene")]),
          "V_eyes_minus_V": diff_ci(rel2[sel("V", "neutral", 1, "gray")], vN)}
    fur_effect = {"V_fur_minus_V_neutral": diff_ci(rel2[sel("V", "fur", 0, "gray")], vN),
                  "O_fur_minus_O_neutral": diff_ci(rel2[sel("O", "fur", 0, "gray")], oN),
                  "O_fur_minus_O_furscr": diff_ci(rel2[sel("O", "fur", 0, "gray")], rel2[sel("O", "fur_scr", 0, "gray")])}

    # P2 population specificity (neutral, no eyes, both backgrounds)
    kV, kO, kU = sel("V", "neutral", 0), sel("O", "neutral", 0), sel("U", "neutral", 0)
    dVO = pop2[kV].mean(0) - pop2[kO].mean(0)
    dVU = pop2[kV].mean(0) - pop2[kU].mean(0)
    rank_VO = int((dVO > dVO[C.UNIT]).sum() + 1)
    rank_VU = int((dVU > dVU[C.UNIT]).sum() + 1)
    P2 = {"delta_VO_unit": float(dVO[C.UNIT]), "rank_VO_of_832": rank_VO, "rank_VU_of_832": rank_VU,
          "delta_VO_percentiles": {q: float(np.percentile(dVO, q)) for q in (50, 90, 95, 99)},
          "HV_pass": bool(rank_VO <= 42)}
    log(f"P1 {P1['V']['mean']:.3f}/{P1['U']['mean']:.3f}/{P1['O']['mean']:.3f}  P2 rank {rank_VO}  ({time.time() - t0:.0f}s)")

    # ============================================================ F2 apertures
    with_head = [i for i, d in enumerate(pets) if d["head"] is not None and
                 max(d["head"][2] - d["head"][0], d["head"][3] - d["head"][1]) >= 40]
    cat_ids = [i for i in with_head if pets[i]["species"] == "cat"]
    dog_ids = [i for i in with_head if pets[i]["species"] == "dog"]
    rng.shuffle(cat_ids); rng.shuffle(dog_ids)
    cat_ids, dog_ids = cat_ids[:300], dog_ids[:300]
    order = np.argsort(-imn_rel)
    top100 = order[:100]
    veh_set = set(g_veh)
    top_veh = [i for i in top100 if imn_cls[i] in veh_set]
    top_anim = [i for i in top100 if imn_cls[i] < 398]
    veh_source = "top100"
    if len(top_veh) < 10:
        top_veh = [i for i in order if imn_cls[i] in veh_set][:30]
        veh_source = "top30_vehicle_class_images (fallback: <10 vehicles in top 100)"
        log("DEVIATION D2: fewer than 10 vehicles in top-100; vehicle aperture content = top 30 vehicle-class images")
    contents = {}
    contents["cat_faces"] = [head_canvas(pets[i], 224, scale_to=SS_) for i in cat_ids]
    contents["dog_faces"] = [head_canvas(pets[i], 224, scale_to=SS_) for i in dog_ids]
    def tr(i):
        iy, ix = S["imn_argmax_yx"][i]
        return translate_to_center(C.to01(C.center_crop_224(C.load_rgb(S["imn_paths"][i]))[0]), iy, ix)
    contents["top_animal"] = [tr(i) for i in top_anim]
    contents["top_vehicle"] = [tr(i) for i in top_veh]
    contents["neutral"] = [C.random_texture(dtd2_all[rng.integers(len(dtd2_all))], rng) for _ in range(200)]
    contents["fur"] = furB[:200]
    masks = {a: C.fill_mask(C.shape_polygon(a, AL, SS_, TH, area=area)) for a in ["V", "U", "O"]}
    f2_imgs, f2_meta = [], []
    for k, lst in contents.items():
        for j, img in enumerate(lst):
            f2_imgs.append(u8(img)); f2_meta.append((k, j, "full"))
            for a, m in masks.items():
                f2_imgs.append(u8(C.composite(C.gray_canvas(), img, m))); f2_meta.append((k, j, a))
    r_f2 = C.run_arrays(probe, f2_imgs, keys={"drive"})
    rel_f2 = r_f2["drive"] / A
    F2 = {}
    tabs = {}
    for k in contents:
        tabs[k] = {a: np.array([rel_f2[i] for i, m in enumerate(f2_meta) if m[0] == k and m[2] == a])
                   for a in ["full", "V", "U", "O"]}
        F2[k] = {a: mean_ci(tabs[k][a]) for a in ["full", "V", "U", "O"]}
        F2[k]["V_minus_O_paired"] = diff_ci(tabs[k]["V"], tabs[k]["O"], paired=True)
        F2[k]["V_minus_U_paired"] = diff_ci(tabs[k]["V"], tabs[k]["U"], paired=True)
    def sigma(kv, ko, nv, no):
        Ssh = 0.5 * ((kv - ko) + (nv - no))
        Cc = 0.5 * ((kv - nv) + (ko - no))
        # D3: negative effects count as zero contribution (S/(S+C) is ill-defined when S < 0)
        Sp, Cp = max(Ssh, 0.0), max(Cc, 0.0)
        if Sp + Cp <= 0:
            return np.nan, Ssh, Cc
        return float(Sp / (Sp + Cp)), Ssh, Cc
    def sigma_boot(k, n=4000):
        kv, ko = tabs[k]["V"], tabs[k]["O"]
        nv, no = tabs["neutral"]["V"], tabs["neutral"]["O"]
        if len(kv) == 0:
            return {"n": 0}
        s0_, Sh, Cc = sigma(kv.mean(), ko.mean(), nv.mean(), no.mean())
        bs = []
        rr = np.random.default_rng(5)
        for _ in range(n):
            i1 = rr.integers(0, len(kv), len(kv)); i2 = rr.integers(0, len(nv), len(nv))
            bs.append(sigma(kv[i1].mean(), ko[i1].mean(), nv[i2].mean(), no[i2].mean())[0])
        bs = np.array(bs, float)
        return {"sigma": s0_, "S_shape_effect": Sh, "C_content_effect": Cc, "n": int(len(kv)),
                "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975))}
    P3 = {k: sigma_boot(k) for k in ["cat_faces", "dog_faces", "top_animal", "top_vehicle", "fur"]}
    P3["vehicle_source"] = veh_source
    log(f"P3 sigma cat {P3['cat_faces'].get('sigma')} veh {P3['top_vehicle'].get('sigma')}  ({time.time() - t0:.0f}s)")

    # ============================================================ F4 routes
    imn_c = S["imn_a4d_at"] * w
    pets_c = S["pets_a4d_at"] * w
    pets_species = S["pets_species"]
    groups_raw = {
        "cats": (np.concatenate([imn_c[np.isin(imn_cls, g_cats)], pets_c[pets_species == "cat"]]),
                 np.concatenate([imn_rel[np.isin(imn_cls, g_cats)], pets_rel[pets_species == "cat"]])),
        "vehicles": (imn_c[np.isin(imn_cls, g_veh)], imn_rel[np.isin(imn_cls, g_veh)]),
        "dogs": (np.concatenate([imn_c[np.isin(imn_cls, C.DOG_IDX)], pets_c[pets_species == "dog"]]),
                 np.concatenate([imn_rel[np.isin(imn_cls, C.DOG_IDX)], pets_rel[pets_species == "dog"]])),
        "wild_canids": (imn_c[np.isin(imn_cls, g_wild)], imn_rel[np.isin(imn_cls, g_wild)]),
        "V_silhouettes": (r2["a4d_at"][kV] * w, rel2[kV]),
    }
    rs = np.random.default_rng(3)
    base_idx = rs.choice(len(imn_c), 500, replace=False)
    baseline = imn_c[base_idx].mean(0)
    def route_analysis(subtract):
        G = {}
        for g, (cv, rl) in groups_raw.items():
            thr = 0.2
            keep = rl >= thr
            if keep.sum() < 30:
                thr = 0.1; keep = rl >= thr
            cvk = cv[keep] - (baseline if subtract else 0)
            cpos = np.clip(cvk, 0, None)
            perm = rs.permutation(len(cpos))
            hA, hB = cpos[perm[: len(perm) // 2]], cpos[perm[len(perm) // 2:]]
            G[g] = {"n": int(len(cpos)), "threshold": thr, "underpowered": bool(len(cpos) < 30), "A": hA, "B": hB}
        out = {}
        for g, d in G.items():
            if len(d["A"]) >= 2 and len(d["B"]) >= 2:
                d["route"] = route(d["A"]); d["routeB"] = route(d["B"])
                d["split_half_J"] = jaccard(d["route"], d["routeB"])
            out[g] = {k: v for k, v in d.items() if k not in ("A", "B", "routeB")}
        pair = {}
        names = [g for g in G if "route" in G[g]]
        for h in names:
            for g in names:
                if g == h:
                    continue
                own = share(G[h]["B"], G[h]["route"])
                pair[f"X_{h}<-{g}"] = share(G[h]["B"], G[g]["route"]) / own if own > 0 else np.nan
                pair[f"J_{h},{g}"] = jaccard(G[h]["route"], G[g]["route"])
        res = {"groups": out, "pairs": pair}
        if "cats" in names and "vehicles" in names:
            ceil = np.mean([G["cats"]["split_half_J"], G["vehicles"]["split_half_J"]])
            J = jaccard(G["cats"]["route"], G["vehicles"]["route"])
            res["J_cat_veh"] = J; res["J_ceiling"] = float(ceil); res["J_rel"] = float(J / ceil) if ceil > 0 else np.nan
            res["X_veh<-cat"] = pair["X_vehicles<-cats"]; res["X_cat<-veh"] = pair["X_cats<-vehicles"]
            if "V_silhouettes" in names:
                res["X_V<-cat"] = pair["X_V_silhouettes<-cats"]; res["X_V<-veh"] = pair["X_V_silhouettes<-vehicles"]
        return res
    P4 = route_analysis(False)
    P4_baseline_sub = route_analysis(True)
    # describe the top channels of each route by their top ImageNetV2 classes
    max4d = S["imn_max4d"]
    def describe(ch):
        top = np.argsort(-max4d[:, ch])[:30]
        cl, cnt = np.unique(imn_cls[top], return_counts=True)
        o = np.argsort(-cnt)[:3]
        return [cats[int(cl[j])] for j in o]
    P4["route_channel_descriptions"] = {g: {int(ch): describe(ch) for ch in d.get("route", [])[:5]}
                                        for g, d in P4["groups"].items()}
    log(f"P4 J_rel {P4.get('J_rel')}  X {P4.get('X_veh<-cat')} {P4.get('X_cat<-veh')}  ({time.time() - t0:.0f}s)")

    # ============================================================ F3 silhouettes
    F3 = f3_silhouettes(probe, S, imn_rel, pets_rel, pets, A, rng, cats)
    log(f"F3 done ({time.time() - t0:.0f}s)")

    # ============================================================ F6 within-category
    def pets_heads(breeds):
        ids = [i for i in with_head if pets[i]["breed"] in breeds]
        imgs = [u8(head_canvas(pets[i], 224, box_frac=1.5)) for i in ids]
        return C.run_arrays(probe, imgs, keys={"drive"})["drive"] / A if imgs else np.array([])
    F6 = {}
    for name, pt, rd in [("pets_cats", C.F6_PETS_CAT_POINTED, C.F6_PETS_CAT_ROUND),
                         ("pets_dogs", C.F6_PETS_DOG_POINTED, C.F6_PETS_DOG_ROUND)]:
        a, b = pets_heads(pt), pets_heads(rd)
        F6[name] = {"pointed": mean_ci(a), "round": mean_ci(b), "pointed_minus_round": diff_ci(a, b),
                    "cohens_d": float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))}
        for br in pt + rd:
            F6[name][f"breed:{br}"] = mean_ci(pets_heads([br]))
    ip = C.resolve_names(C.F6_IMNV2_POINTED, cats, []); ir = C.resolve_names(C.F6_IMNV2_ROUND, cats, [])
    a, b = imn_rel[np.isin(imn_cls, ip)], imn_rel[np.isin(imn_cls, ir)]
    F6["imagenetv2"] = {"pointed": mean_ci(a), "round": mean_ci(b), "pointed_minus_round": diff_ci(a, b),
                        "cohens_d": float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)),
                        "Siamese_minus_Persian": diff_ci(imn_rel[imn_cls == cats.index("Siamese cat")],
                                                         imn_rel[imn_cls == cats.index("Persian cat")])}

    # ============================================================ F7 objects
    cls_mean = np.array([imn_rel[imn_cls == c].mean() for c in range(1000)])
    cls_pre = np.array([S["imn_pre_mean"][imn_cls == c].mean() for c in range(1000)])
    F7 = {"all_class_median_rel": float(np.median(cls_mean)),
          "all_image_median_pre_mean": float(np.median(S["imn_pre_mean"]))}
    for name, lst in [("pointed", C.F7_POINTED), ("round", C.F7_ROUND), ("essay_examples", C.F7_ESSAY)]:
        idx = C.resolve_names(lst, cats, LOG)
        F7[name] = {"classes": {cats[c]: {"rel": float(cls_mean[c]), "pre_mean": float(cls_pre[c]),
                                          "class_rank_of_1000": int((cls_mean > cls_mean[c]).sum() + 1)} for c in idx},
                    "mean_rel": float(cls_mean[idx].mean()), "mean_pre_mean": float(cls_pre[idx].mean())}
    F7["pointed_minus_round_rel"] = diff_ci(cls_mean[C.resolve_names(C.F7_POINTED, cats, [])],
                                            cls_mean[C.resolve_names(C.F7_ROUND, cats, [])])
    top_classes = np.argsort(-cls_mean)[:20]
    F7["top20_classes"] = [(cats[c], float(cls_mean[c])) for c in top_classes]

    # ============================================================ decisions
    def g(d, *ks):
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return np.nan
            d = d[k]
        return d
    s_cat, s_veh = g(P3, "cat_faces", "sigma"), g(P3, "top_vehicle", "sigma")
    eye = g(P5, "O_eyes_minus_O", "diff")
    Jrel, Xvc, Xcv = P4.get("J_rel", np.nan), P4.get("X_veh<-cat", np.nan), P4.get("X_cat<-veh", np.nan)
    XVc, XVv = P4.get("X_V<-cat", np.nan), P4.get("X_V<-veh", np.nan)
    shared = bool(Jrel >= 0.5 and Xvc >= 0.5 and Xcv >= 0.5)
    disjoint = bool(Jrel <= 0.2 and Xvc <= 0.25 and Xcv <= 0.25)
    v1 = st1["V1_pass"]
    dec = {
        "P1_HV_pass": P1["HV_pass"], "P2_HV_pass": P2["HV_pass"],
        "P3_sigma_cat": s_cat, "P3_sigma_vehicle": s_veh,
        "P3_HV_pass": bool(s_cat >= 0.67 and s_veh >= 0.67),
        "P3_low_both": bool(s_cat <= 0.33 and s_veh <= 0.33),
        "P4_shared_route": shared, "P4_disjoint_routes": disjoint,
        "P4_V_carried_by_cat_and_vehicle_routes": bool(XVc >= 0.5 and XVv >= 0.5),
        "P5_eye_effect": eye,
        "H_V_supported": bool(P1["HV_pass"] and P2["HV_pass"] and s_cat >= 0.67 and s_veh >= 0.67 and shared
                              and XVc >= 0.5 and XVv >= 0.5),
        "H_V_refuted": bool(v1 and not P1["HV_pass"] and s_cat <= 0.33 and s_veh <= 0.33),
        "H_OR_supported": bool(not P1["HV_pass"] and s_cat <= 0.33 and s_veh <= 0.33 and disjoint and eye < 0.15),
        "H_face_supported": bool(not P1["HV_pass"] and eye >= 0.15 and Jrel >= 0.5 and XVc < 0.5 and XVv < 0.5),
    }
    out = {"unit": C.UNIT, "A_ref": A, "theta_star": TH, "alpha_star": AL, "s_star": SS_,
           "P1": P1, "P2": P2, "P3": P3, "P4": P4, "P4_baseline_subtracted": P4_baseline_sub, "P5": P5,
           "fur_effects": fur_effect, "stage2_cells": cell_table, "F2_apertures": F2, "F3": F3, "F6": F6,
           "F7": F7, "decisions": dec, "log": LOG, "runtime_s": time.time() - t0}
    C.save_json(out, f"stage2_tests{args.tag}.json")
    log(json.dumps(dec, indent=1, default=str))

    # example figures
    sheet = Image.new("RGB", (8 * 112, 3 * 112), (255, 255, 255))
    ex = [sel("V", "neutral", 0, "gray"), sel("U", "neutral", 0, "gray"), sel("O", "neutral", 0, "gray"),
          sel("O", "neutral", 1, "gray"), sel("V", "fur", 0, "scene"), sel("O", "fur", 1, "scene"),
          sel("V", "fur_scr", 0, "gray"), sel("U", "fur", 1, "gray")]
    for j, k in enumerate(ex):
        sheet.paste(Image.fromarray(stims[int(np.where(k)[0][0])]).resize((112, 112)), (j * 112, 0))
    for row, kname in [(1, "cat_faces"), (2, "top_animal")]:
        ids = [i for i, m in enumerate(f2_meta) if m[0] == kname and m[1] in (0, 1)]
        for j, i in enumerate(ids[:8]):
            sheet.paste(Image.fromarray(f2_imgs[i]).resize((112, 112)), (j * 112, row * 112))
    sheet.save(os.path.join(C.RESULTS, f"fig_stage2_examples{args.tag}.jpg"), quality=88)


def f3_silhouettes(probe, S, imn_rel, pets_rel, pets, A, rng, cats):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    det = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    items = []
    # ImageNetV2 top 100 with Mask R-CNN masks
    excl = 0
    for i in np.argsort(-imn_rel)[:100]:
        img = C.to01(C.center_crop_224(C.load_rgb(S["imn_paths"][i]))[0])
        iy, ix = S["imn_argmax_yx"][i]
        py, px = 16 * iy + 8, 16 * ix + 8
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
            excl += 1
            continue
        items.append(("imagenetv2", img, best[1].astype(np.float32)))
    # Pets top 100 with trimaps
    for i in np.argsort(-pets_rel)[:100]:
        d = pets[i]
        img_pil, (s, l, t) = C.center_crop_224(C.load_rgb(d["path"]))
        tri = Image.open(d["trimap"])
        w, h = tri.size
        tri = tri.resize((max(224, round(w * s)), max(224, round(h * s))), Image.NEAREST).crop((l, t, l + 224, t + 224))
        tri = np.asarray(tri)
        items.append(("pets", C.to01(img_pil), ((tri == 1) | (tri == 3)).astype(np.float32)))
    imgs, meta = [], []
    for j, (src, img, m) in enumerate(items):
        mo = img[m > 0.5].mean(0) if (m > 0.5).any() else C.GRAY
        mb = img[m < 0.5].mean(0) if (m < 0.5).any() else C.GRAY
        obj_only = img * m[..., None] + mo * (1 - m[..., None])
        bg_only = img * (1 - m[..., None]) + mb * m[..., None]
        edits = {
            "orig": img,
            "sil_flat": C.composite(img, np.broadcast_to(mo, img.shape), m),
            "sil_scr": C.composite(img, C.phase_scramble(obj_only, rng), m),
            "bg_flat": C.composite(np.broadcast_to(mb, img.shape).copy(), img, m),
            "bg_scr": C.composite(C.phase_scramble(bg_only, rng), img, m),
        }
        for k, v in edits.items():
            imgs.append(u8(v)); meta.append((src, j, k))
    r = C.run_arrays(probe, imgs, keys={"drive"})
    rel = r["drive"] / A
    res = {"n_imagenetv2": sum(1 for x in items if x[0] == "imagenetv2"), "excluded_no_mask": excl,
           "n_pets": sum(1 for x in items if x[0] == "pets")}
    for src in ["imagenetv2", "pets", "all"]:
        res[src] = {}
        ids = sorted({m[1] for m in meta if src == "all" or m[0] == src})
        orig = np.array([rel[[i for i, m in enumerate(meta) if m[1] == j and m[2] == "orig"][0]] for j in ids])
        for k in ["sil_flat", "sil_scr", "bg_flat", "bg_scr"]:
            ed = np.array([rel[[i for i, m in enumerate(meta) if m[1] == j and m[2] == k][0]] for j in ids])
            R = ed / np.maximum(orig, 1e-6)
            res[src][k] = {"R_mean": mean_ci(R), "R_median": float(np.median(R)), "rel_edit": mean_ci(ed)}
        res[src]["orig_rel"] = mean_ci(orig)
    # example sheet
    sheet = Image.new("RGB", (5 * 112, 4 * 112))
    for row, j in enumerate([0, 1, res["n_imagenetv2"], res["n_imagenetv2"] + 1]):
        for col, k in enumerate(["orig", "sil_flat", "sil_scr", "bg_flat", "bg_scr"]):
            i = [ii for ii, m in enumerate(meta) if m[1] == j and m[2] == k][0]
            sheet.paste(Image.fromarray(imgs[i]).resize((112, 112)), (col * 112, row * 112))
    sheet.save(os.path.join(C.RESULTS, f"fig_F3_examples{args.tag}.jpg"), quality=88)
    return res


if __name__ == "__main__":
    main()
