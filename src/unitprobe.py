"""Generic probe for a unit that is a 1x1 conv over some hooked input (study 2: mixed4d pool branch)."""
import numpy as np
import torch

import common as C

# mixed4d concat = [1x1 (112), 3x3 (288), 5x5 (64), pool_reduce (64)] -> pool branch starts at 464
SPEC_4D_POOL = dict(layer="mixed4d", pre_module="mixed4d_pool_reduce_pre_relu_conv", pre_offset=464,
                    route_module="mixed4d_pool", desc_layer="mixed4c")


class UnitProbe:
    def __init__(self, ch, spec=SPEC_4D_POOL):
        from lucent.modelzoo import inceptionv1
        m = inceptionv1(pretrained=True, redirected_ReLU=False).eval().to(C.DEVICE)
        self.model, self.ch, self.spec = m, ch, spec
        self.pre_ch = ch - spec["pre_offset"]
        self.acts = {}
        for name in {spec["layer"], spec["pre_module"], spec["route_module"], spec["desc_layer"]}:
            getattr(m, name).register_forward_hook(self._hook(name))
        conv = getattr(m, spec["pre_module"])
        self.w = conv.weight.detach()[self.pre_ch, :, 0, 0].float().cpu().numpy()
        self.b = float(conv.bias.detach()[self.pre_ch].cpu())

    def _hook(self, name):
        def fn(_m, _i, out):
            self.acts[name] = out.detach()
        return fn

    @torch.no_grad()
    def run(self, batch01):
        x = batch01.to(C.DEVICE) * 255.0 - 117.0
        self.model(x)
        s = self.spec
        lay = self.acts[s["layer"]].float()
        post = lay[:, self.ch]
        pre = self.acts[s["pre_module"]][:, self.pre_ch].float()
        rin = self.acts[s["route_module"]].float()
        desc = self.acts[s["desc_layer"]].float()
        B = post.shape[0]
        flat = post.reshape(B, -1)
        amax = flat.argmax(1)
        iy, ix = amax // post.shape[2], amax % post.shape[2]
        ar = torch.arange(B)
        return {
            "drive": flat.max(1).values.cpu().numpy(),
            "pre_mean": pre.mean(dim=(1, 2)).cpu().numpy(),
            "pre_at": pre[ar, iy, ix].cpu().numpy(),
            "argmax_yx": torch.stack([iy, ix], 1).cpu().numpy(),
            "route_at": rin[ar, :, iy, ix].cpu().numpy(),
            "max_pop": lay.amax(dim=(2, 3)).cpu().numpy(),
            "max_desc": desc.amax(dim=(2, 3)).cpu().numpy(),
            "post_equals_relu_pre_err": float((post - pre.clamp(min=0)).abs().max().cpu()),
        }


def run_arrays(probe, arrays, bs=64, keys=None):
    res = {}
    for s in range(0, len(arrays), bs):
        chunk = np.stack(arrays[s:s + bs])
        chunk = chunk.astype(np.float32) / 255.0 if chunk.dtype == np.uint8 else chunk.astype(np.float32)
        o = probe.run(torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous())
        for k, v in o.items():
            if keys is None or k in keys:
                res.setdefault(k, []).append(np.atleast_1d(v))
    return {k: np.concatenate(v) for k, v in res.items()}
