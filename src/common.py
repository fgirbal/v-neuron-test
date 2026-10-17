"""Shared model, data and stimulus utilities for the mixed4e:55 V-neuron test.

Run scripts with `python -I`; each script adds this directory to sys.path itself.
"""
import os
import glob
import json
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageDraw

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
SCR = os.environ.get("VNT_SCRATCH", os.path.join(_ROOT, "scratch"))  # datasets, weights, scan caches
PROJ = os.environ.get("VNT_PROJECT", _ROOT)  # results folders are written here
DATA = os.path.join(SCR, "data")
RESULTS = os.path.join(PROJ, "results")
WORK = os.path.join(SCR, "work")  # large scan arrays (.npz)
os.makedirs(WORK, exist_ok=True)
IMNV2_DIR = os.path.join(DATA, "imagenetv2", "imagenetv2-matched-frequency-format-val")
PETS_DIR = os.path.join(DATA, "pets")
DTD_DIR = os.path.join(DATA, "dtd", "dtd", "images")
os.environ.setdefault("TORCH_HOME", os.path.join(SCR, "cache"))

UNIT = 55
CANVAS = 224
GRAY = np.array([124, 116, 104], dtype=np.float32) / 255.0  # ImageNet mean colour
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CURVE_UNITS_3B = [379, 385, 343, 342, 388, 324, 340, 330, 349, 406]

# ---------------------------------------------------------------- class groups
def imagenet_categories():
    from torchvision.models import ResNet50_Weights
    return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])

CATS = ["tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat"]
WILD_CANIDS = ["red fox", "kit fox", "Arctic fox", "grey fox", "timber wolf", "white wolf",
               "red wolf", "coyote", "dingo", "dhole", "African hunting dog"]
VEHICLES = ["sports car", "convertible", "cab", "racer", "limousine", "minivan", "beach wagon",
            "Model T", "jeep", "police van", "ambulance", "fire engine", "garbage truck", "pickup",
            "tow truck", "trailer truck", "moving van", "minibus", "school bus", "trolleybus", "grille"]
DOG_IDX = list(range(151, 269))

F6_IMNV2_POINTED = ["borzoi", "collie", "Shetland sheepdog", "Afghan hound", "Saluki", "whippet",
                    "Ibizan hound", "basenji", "Siamese cat", "red fox", "kit fox", "Arctic fox", "grey fox"]
F6_IMNV2_ROUND = ["pug", "Pekinese", "Shih-Tzu", "boxer", "French bulldog", "bull mastiff",
                  "Japanese spaniel", "Persian cat"]
F6_PETS_CAT_POINTED = ["Siamese", "Abyssinian", "Sphynx"]
F6_PETS_CAT_ROUND = ["Persian", "British_Shorthair", "Bombay"]
F6_PETS_DOG_POINTED = ["shiba_inu", "miniature_pinscher", "keeshond", "samoyed"]
F6_PETS_DOG_ROUND = ["pug", "japanese_chin", "boxer", "american_bulldog"]

F7_POINTED = ["pick", "missile", "projectile", "warplane", "space shuttle", "speedboat", "canoe",
              "bell cote", "volcano", "alp"]
F7_ROUND = ["golf ball", "tennis ball", "soccer ball", "basketball", "volleyball", "ping-pong ball",
            "orange", "wall clock", "analog clock", "plate", "frying pan", "manhole cover", "airship",
            "bubble", "balloon"]
F7_ESSAY = ["bullet train", "pool table", "fireboat"]

SCENE_CLASSES = ["valley", "lakeside", "seashore", "sandbar", "cliff", "promontory"]
DTD_STAGE1 = ["marbled", "stratified", "grooved", "wrinkled"]
DTD_STAGE2 = ["blotchy", "cracked", "crystalline", "flecked", "pleated", "smeared", "stained", "veined"]


def resolve_names(names, cats, log):
    idx = []
    for n in names:
        if n in cats:
            idx.append(cats.index(n))
        else:
            log.append(f"class name not found, dropped: {n!r}")
    return idx

# ---------------------------------------------------------------- model
class Probe:
    """InceptionV1 (Lucid weights) with hooks on the layers we read."""

    def __init__(self):
        from lucent.modelzoo import inceptionv1
        m = inceptionv1(pretrained=True, redirected_ReLU=False).eval().to(DEVICE)
        self.model = m
        self.acts = {}
        for name in ["mixed3b", "mixed4d", "mixed4e", "mixed4e_1x1_pre_relu_conv"]:
            getattr(m, name).register_forward_hook(self._hook(name))
        w = m.mixed4e_1x1_pre_relu_conv.weight.detach()[:, :, 0, 0]  # [256, 528]
        self.w55 = w[UNIT].float().cpu().numpy()
        self.b55 = float(m.mixed4e_1x1_pre_relu_conv.bias.detach()[UNIT].cpu())

    def _hook(self, name):
        def fn(_m, _i, out):
            self.acts[name] = out.detach()
        return fn

    @torch.no_grad()
    def run(self, batch01):
        """batch01: float tensor [B,3,224,224] in [0,1]. Returns dict of numpy arrays."""
        x = batch01.to(DEVICE) * 255.0 - 117.0
        self.model(x)
        a4e = self.acts["mixed4e"].float()                  # [B,832,14,14] post-ReLU
        pre = self.acts["mixed4e_1x1_pre_relu_conv"][:, UNIT].float()  # [B,14,14]
        a4d = self.acts["mixed4d"].float()                  # [B,528,14,14]
        a3b = self.acts["mixed3b"].float()                  # [B,480,28,28]
        B = a4e.shape[0]
        post55 = a4e[:, UNIT]
        flat = post55.reshape(B, -1)
        amax = flat.argmax(1)
        iy, ix = amax // post55.shape[2], amax % post55.shape[2]
        # mixed4d at the unit's argmax location (for exact 1x1 route decomposition)
        a4d_at = a4d[torch.arange(B), :, iy, ix]            # [B,528]
        pre_at = pre[torch.arange(B), iy, ix]
        out = {
            "max4e": a4e.amax(dim=(2, 3)).cpu().numpy(),     # [B,832]
            "max4d": a4d.amax(dim=(2, 3)).cpu().numpy(),     # [B,528]
            "max3b": a3b.amax(dim=(2, 3)).cpu().numpy(),     # [B,480]
            "center3b": a3b[:, :, 14, 14].cpu().numpy(),    # [B,480]
            "drive": flat.max(1).values.cpu().numpy(),       # post-ReLU spatial max of unit
            "pre_mean": pre.mean(dim=(1, 2)).cpu().numpy(),  # negative-pole measure
            "pre_max": pre.reshape(B, -1).max(1).values.cpu().numpy(),
            "pre_at": pre_at.cpu().numpy(),
            "argmax_yx": torch.stack([iy, ix], 1).cpu().numpy(),
            "a4d_at": a4d_at.cpu().numpy(),
        }
        return out


def run_arrays(probe, arrays, bs=64, keys=None):
    """arrays: list/array of HxWx3 float images in [0,1] (224x224). Concatenates outputs."""
    res = {}
    for s in range(0, len(arrays), bs):
        chunk = np.stack(arrays[s:s + bs])
        chunk = chunk.astype(np.float32) / 255.0 if chunk.dtype == np.uint8 else chunk.astype(np.float32)
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
        o = probe.run(t)
        for k, v in o.items():
            if keys is None or k in keys:
                res.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in res.items()}

# ---------------------------------------------------------------- images
def load_rgb(path):
    return Image.open(path).convert("RGB")


def center_crop_224(img):
    w, h = img.size
    s = CANVAS / min(w, h)
    img = img.resize((max(CANVAS, round(w * s)), max(CANVAS, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - CANVAS) // 2, (h - CANVAS) // 2
    return img.crop((l, t, l + CANVAS, t + CANVAS)), (s, l, t)


def to01(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def imagenetv2_index():
    items = []
    for cls in range(1000):
        for p in sorted(glob.glob(os.path.join(IMNV2_DIR, str(cls), "*"))):
            items.append((p, cls))
    return items


def pets_index():
    """Returns list of dicts: path, breed, species ('cat'/'dog'), trimap path, head box (or None)."""
    ann = os.path.join(PETS_DIR, "annotations")
    out = []
    with open(os.path.join(ann, "list.txt")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            name, _cid, species, _bid = line.split()
            breed = name.rsplit("_", 1)[0]
            head = None
            xp = os.path.join(ann, "xmls", name + ".xml")
            if os.path.exists(xp):
                r = ET.parse(xp).getroot()
                bb = r.find("object/bndbox")
                if bb is not None:
                    head = [int(float(bb.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
            out.append({"path": os.path.join(PETS_DIR, "images", name + ".jpg"),
                        "trimap": os.path.join(ann, "trimaps", name + ".png"),
                        "breed": breed, "species": "cat" if species == "1" else "dog",
                        "head": head, "name": name})
    return out


def dtd_paths(categories):
    out = {}
    for c in categories:
        out[c] = sorted(glob.glob(os.path.join(DTD_DIR, c, "*.jpg")))
    return out


def random_texture(path, rng, size=CANVAS):
    img = load_rgb(path)
    w, h = img.size
    c = int(min(w, h) * rng.uniform(0.5, 1.0))
    l = rng.integers(0, w - c + 1)
    t = rng.integers(0, h - c + 1)
    img = img.crop((l, t, l + c, t + c)).resize((size, size), Image.BICUBIC)
    return to01(img)


def phase_scramble(img01, rng):
    """Randomise Fourier phase (same phase field for all channels); keeps amplitude spectrum."""
    h, w, _ = img01.shape
    rand_phase = np.angle(np.fft.fft2(rng.random((h, w))))
    out = np.empty_like(img01)
    for c in range(3):
        f = np.fft.fft2(img01[:, :, c])
        out[:, :, c] = np.real(np.fft.ifft2(np.abs(f) * np.exp(1j * (np.angle(f) + rand_phase))))
    return np.clip(out, 0, 1)

# ---------------------------------------------------------------- shapes
SS = 4  # supersampling for anti-aliased masks


def _rot(pts, theta_deg, center):
    t = np.deg2rad(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (pts - center) @ R.T + center


def _polygon_area_centroid(p):
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    cx = ((x + x1) * cross).sum() / (6 * a)
    cy = ((y + y1) * cross).sum() / (6 * a)
    return abs(a), np.array([cx, cy])


def v_polygon(alpha, size):
    """Filled V = isosceles triangle, apex pointing down (+y), apex angle alpha. max(base,height)=size."""
    a = np.deg2rad(alpha) / 2
    base_over_L, h_over_L = 2 * np.sin(a), np.cos(a)
    L = size / max(base_over_L, h_over_L)
    apex = np.array([0.0, 0.0])
    left = np.array([-L * np.sin(a), -L * np.cos(a)])
    right = np.array([L * np.sin(a), -L * np.cos(a)])
    return np.stack([left, right, apex]), L


def u_polygon(alpha, size, r_frac=0.35, n_arc=40):
    """Same arms as the V, apex replaced by a circular arc of radius r_frac*L tangent to both arms."""
    a = np.deg2rad(alpha) / 2
    tri, L = v_polygon(alpha, size)
    r = r_frac * L
    d = r / np.sin(a)                      # apex -> arc centre distance along the axis
    c = np.array([0.0, -d])
    # tangent points sit at angles (pi - a) and a around the centre (+y is down); the arc
    # between them passes through the bottom point at angle pi/2
    angs = np.linspace(np.pi - a, a, n_arc)
    arc = np.stack([c[0] + r * np.cos(angs), c[1] + r * np.sin(angs)], 1)
    return np.concatenate([[tri[0]], arc, [tri[1]]], 0)


def shape_polygon(shape, alpha, size, theta, center=(CANVAS / 2, CANVAS / 2), area=None):
    """Polygon (N,2) for 'V','U','O' with centroid at `center`, rotated by theta (0 = apex down).
    If `area` is given, the polygon is scaled to that area."""
    if shape == "V":
        p, _ = v_polygon(alpha, size)
    elif shape == "U":
        p = u_polygon(alpha, size)
    elif shape == "O":
        r = size / 2
        ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        p = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    else:
        raise ValueError(shape)
    A, c = _polygon_area_centroid(p)
    p = p - c
    if area is not None:
        p = p * np.sqrt(area / A)
    p = p + np.asarray(center)
    return _rot(p, theta, np.asarray(center))


def polygon_area(shape, alpha, size):
    p = shape_polygon(shape, alpha, size, 0)
    return _polygon_area_centroid(p)[0]


def fill_mask(poly, canvas=CANVAS):
    im = Image.new("L", (canvas * SS, canvas * SS), 0)
    ImageDraw.Draw(im).polygon([tuple(q) for q in (poly * SS)], fill=255)
    im = im.resize((canvas, canvas), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def stroke_mask(polyline, width, closed=False, canvas=CANVAS):
    im = Image.new("L", (canvas * SS, canvas * SS), 0)
    pts = [tuple(q) for q in (polyline * SS)]
    if closed:
        pts = pts + [pts[0]]
    ImageDraw.Draw(im).line(pts, fill=255, width=int(width * SS), joint="curve")
    im = im.resize((canvas, canvas), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def outline_polyline(shape, alpha, size, theta, center=(CANVAS / 2, CANVAS / 2)):
    """Open V/U outlines (no top edge), closed circle for O, straight bar."""
    c = np.asarray(center)
    if shape == "V":
        p, _ = v_polygon(alpha, size)
        line = np.stack([p[0], p[2], p[1]])
    elif shape == "U":
        line = u_polygon(alpha, size)
    elif shape == "O":
        r = size / 2
        ang = np.linspace(0, 2 * np.pi, 97)
        line = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    elif shape == "bar":
        line = np.array([[0, -size / 2], [0, size / 2]], dtype=float)
    else:
        raise ValueError(shape)
    # centre on bounding-box centre for outlines
    mid = (line.max(0) + line.min(0)) / 2
    line = line - mid + c
    return _rot(line, theta, c)


def eyes_layer(shape, alpha, size, theta, area, center=(CANVAS / 2, CANVAS / 2)):
    """Two dark disks with highlights in the upper interior of the shape (in its own frame).
    Returns (alpha_mask, rgb) to composite on top."""
    c = np.asarray(center)
    poly0 = shape_polygon(shape, alpha, size, 0, center=center, area=area)
    ys = poly0[:, 1]
    top, bot = ys.min(), ys.max()
    ye = top + 0.30 * (bot - top)
    # width of the shape at height ye (unrotated)
    m = fill_mask(poly0)
    row = m[int(round(ye))]
    xs = np.where(row > 0.5)[0]
    if len(xs) < 4:
        return np.zeros((CANVAS, CANVAS), np.float32), None
    xl, xr = xs.min(), xs.max()
    width = xr - xl
    xm = (xl + xr) / 2
    sep = 0.45 * width
    rad = min(0.07 * size, 0.14 * width)
    eye_a = np.zeros((CANVAS * SS, CANVAS * SS), np.uint8)
    hi_a = np.zeros_like(eye_a)
    imE, imH = Image.fromarray(eye_a), Image.fromarray(hi_a)
    dE, dH = ImageDraw.Draw(imE), ImageDraw.Draw(imH)
    for sx in (-1, 1):
        e = np.array([xm + sx * sep / 2, ye])
        e = _rot(e[None], theta, c)[0] * SS
        R = rad * SS
        dE.ellipse([e[0] - R, e[1] - R, e[0] + R, e[1] + R], fill=255)
        h = e + np.array([-0.35 * R, -0.35 * R])
        rh = 0.3 * R
        dH.ellipse([h[0] - rh, h[1] - rh, h[0] + rh, h[1] + rh], fill=255)
    E = np.asarray(imE.resize((CANVAS, CANVAS), Image.LANCZOS), np.float32) / 255
    H = np.asarray(imH.resize((CANVAS, CANVAS), Image.LANCZOS), np.float32) / 255
    return E, H


def composite(bg, fg, mask):
    return bg * (1 - mask[..., None]) + fg * mask[..., None]


def gray_canvas():
    return np.tile(GRAY, (CANVAS, CANVAS, 1)).astype(np.float32)

# ---------------------------------------------------------------- stats
def boot_ci(x, fn=np.mean, n=10000, seed=0, alpha=0.05):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    stats = np.array([fn(x[i]) for i in idx]) if fn is not np.mean else x[idx].mean(1)
    return float(fn(x)), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def boot_diff(a, b, n=10000, seed=0, alpha=0.05):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def save_json(obj, name):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
