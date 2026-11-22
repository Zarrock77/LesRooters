"""Round-trip fresh watermarked images through latent-diffusion VAEs (generator environment).

Reads <dir>/fresh_<method>.npz (arrays S, W, bits) and writes <dir>/extra_<method>.npz with one array per VAE.
The 16-channel FLUX-family VAE is tried from several public repositories that ship it.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

folder = Path(sys.argv[1])
FLUX_CANDIDATES = [("ostris/Flex.1-alpha", "vae"), ("Alpha-VLLM/Lumina-Image-2.0", "vae"), ("Tongyi-MAI/Z-Image-Turbo", "vae"),
                   ("HiDream-ai/HiDream-I1-Full", "vae"), ("Shakker-Labs/AWPortrait-FL", "vae"), ("black-forest-labs/FLUX.1-schnell", "vae")]


def load_flux():
    for repo, sub in FLUX_CANDIDATES:
        try:
            vae = AutoencoderKL.from_pretrained(repo, subfolder=sub, torch_dtype=torch.float32)
            print("FLUX_VAE_FROM", repo, "latent channels", vae.config.latent_channels, flush=True)
            return vae
        except Exception as error:
            print("FLUX_VAE_SKIP", repo, repr(error)[:120], flush=True)
    return None


out = {m: {} for m in ("pixelseal", "trustmark", "stegastamp")}
vae = load_flux()
if vae is not None:
    vae = vae.cuda().eval()
    for method in out:
        with np.load(folder / f"fresh_{method}.npz") as z:
            W = z["W"]
        res = []
        with torch.no_grad():
            for s in range(0, len(W), 8):
                x = torch.from_numpy(W[s:s + 8]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
                y = vae.decode(vae.encode(x).latent_dist.mode()).sample
                res.append(((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        out[method]["vae_flux"] = np.concatenate(res)
    print("VAE_DONE vae_flux", flush=True)
for method, arrays in out.items():
    if arrays:
        np.savez_compressed(folder / f"extra_{method}.npz", **arrays)
print("ROUNDTRIP_DONE", flush=True)
