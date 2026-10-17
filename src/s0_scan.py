"""Stage 0: scan ImageNetV2 and Oxford Pets, compute A_ref, run validity check V0."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import common as C


def load_crop(path):
    img, _ = C.center_crop_224(C.load_rgb(path))
    return C.to01(img)


def scan(probe, paths, label):
    t0 = time.time()
    outs = {}
    bs = 64
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(paths), bs):
            arrs = list(ex.map(load_crop, paths[s:s + bs]))
            o = C.run_arrays(probe, arrs, bs=bs)
            for k, v in o.items():
                outs.setdefault(k, []).append(v)
            if s % 1280 == 0:
                print(f"{label}: {s}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
    return {k: np.concatenate(v) for k, v in outs.items()}


def main():
    probe = C.Probe()
    cats = C.imagenet_categories()

    items = C.imagenetv2_index()
    paths = [p for p, _ in items]
    cls = np.array([c for _, c in items])
    r = scan(probe, paths, "imagenetv2")
    np.savez(os.path.join(C.WORK, "scan_imnv2.npz"), cls=cls, paths=np.array(paths), **r)

    pets = C.pets_index()
    ppaths = [d["path"] for d in pets]
    rp = scan(probe, ppaths, "pets")
    np.savez(os.path.join(C.WORK, "scan_pets.npz"), paths=np.array(ppaths),
             breed=np.array([d["breed"] for d in pets]), species=np.array([d["species"] for d in pets]), **rp)

    # A_ref: mean of top 0.2% (top 20 of 10,000) of ImageNetV2, per channel
    k = int(round(0.002 * len(paths)))
    aref4e = np.sort(r["max4e"], 0)[-k:].mean(0)
    aref3b = np.sort(r["max3b"], 0)[-k:].mean(0)
    aref4d = np.sort(r["max4d"], 0)[-k:].mean(0)
    np.savez(os.path.join(C.WORK, "aref.npz"), aref4e=aref4e, aref3b=aref3b, aref4d=aref4d)

    # exactness of the 1x1 decomposition
    recon = C.np.asarray(r["a4d_at"]) @ probe.w55 + probe.b55
    err = float(np.abs(recon - r["pre_at"]).max())

    # V0: unit identity
    log = []
    grp = {
        "cats": set(C.resolve_names(C.CATS, cats, log)),
        "wild_canids": set(C.resolve_names(C.WILD_CANIDS, cats, log)),
        "dogs": set(C.DOG_IDX),
        "vehicles": set(C.resolve_names(C.VEHICLES, cats, log)),
    }
    order = np.argsort(-r["drive"])
    top50 = order[:50]
    def g_of(c):
        for g, s in grp.items():
            if c in s:
                return g
        return "other"
    groups = [g_of(int(cls[i])) for i in top50]
    counts = {g: groups.count(g) for g in ["cats", "wild_canids", "dogs", "vehicles", "other"]}
    in_any = 50 - counts["other"]
    v0 = {
        "A_ref_55": float(aref4e[C.UNIT]),
        "top50": [{"rank": j + 1, "class": cats[int(cls[i])], "rel": float(r["drive"][i] / aref4e[C.UNIT]),
                   "path": os.path.basename(paths[i])} for j, i in enumerate(top50)],
        "counts": counts,
        "frac_in_groups": in_any / 50,
        "criterion": ">=60% in cats|wild canids|dogs|vehicles, >=5 cats+wild canids, >=3 vehicles",
        "pass": bool(in_any / 50 >= 0.6 and counts["cats"] + counts["wild_canids"] >= 5 and counts["vehicles"] >= 3),
        "decomposition_max_abs_error": err,
        "pre_at_scale": float(np.abs(r["pre_at"]).max()),
        "log": log,
    }
    C.save_json(v0, "v0_unit_identity.json")
    print("V0 pass:", v0["pass"], counts, "decomp err", err)

    # contact sheet of top 24 ImageNetV2 images
    sheet = Image.new("RGB", (6 * 112, 4 * 112))
    for j, i in enumerate(order[:24]):
        im = Image.fromarray((load_crop(paths[i]) * 255).astype(np.uint8)).resize((112, 112))
        sheet.paste(im, ((j % 6) * 112, (j // 6) * 112))
    sheet.save(os.path.join(C.RESULTS, "fig_v0_top24_imagenetv2.jpg"), quality=88)


if __name__ == "__main__":
    main()
