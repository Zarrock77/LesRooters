"""Re-render fresh watermarked pictures with the 16-channel FLUX-family VAE, with repeats and latent noise
(generator environment). Emulates an editing model that outputs the whole picture instead of pasting the
untouched pixels back.

Reads <dir>/fresh_<method>.npz (arrays S, W, bits), writes <dir>/rerender_<method>.npz (one array per variant).
usage: rerender.py <dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

folder = Path(sys.argv[1])
CANDIDATES = [("ostris/Flex.1-alpha", "vae"), ("black-forest-labs/FLUX.1-schnell", "vae"), ("Alpha-VLLM/Lumina-Image-2.0", "vae")]
vae = None
for repo, sub in CANDIDATES:
    try:
        vae = AutoencoderKL.from_pretrained(repo, subfolder=sub, torch_dtype=torch.float32).cuda().eval()
        print("VAE_FROM", repo, flush=True)
        break
    except Exception as error:  # noqa: BLE001
        print("VAE_SKIP", repo, repr(error)[:100], flush=True)
assert vae is not None
gen = torch.Generator(device="cuda").manual_seed(0)


@torch.no_grad()
def render(x, passes=1, noise=0.0, sample=False):
    for _ in range(passes):
        dist = vae.encode(x).latent_dist
        z = dist.sample(generator=gen) if sample else dist.mode()
        if noise:
            z = z + noise * z.std() * torch.randn(z.shape, device=z.device, generator=gen)
        x = vae.decode(z).sample.clamp(-1, 1)
        x = ((x + 1) * 127.5).round() / 127.5 - 1            # the result is stored as an 8 bit picture each time
    return x


VARIANTS = {"vae1": {}, "vae2": {"passes": 2}, "vae3": {"passes": 3}, "vae_sample": {"sample": True},
            "vae_n0.05": {"noise": 0.05}, "vae_n0.1": {"noise": 0.1}, "vae_n0.2": {"noise": 0.2}, "vae_n0.35": {"noise": 0.35}}
for method in ("pixelseal", "trustmark", "stegastamp"):
    with np.load(folder / f"fresh_{method}.npz") as z:
        W = z["W"][:24]
    out = {}
    for name, kw in VARIANTS.items():
        res = []
        for s in range(0, len(W), 8):
            x = torch.from_numpy(W[s:s + 8]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            res.append(((render(x, **kw) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        out[name] = np.concatenate(res)
    np.savez_compressed(folder / f"rerender_{method}.npz", **out)
    print("RERENDER", method, len(W), list(out), flush=True)
print("RERENDER_DONE", flush=True)
