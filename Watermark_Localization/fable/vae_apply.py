"""Round-trip an image array [N,512,512,3] uint8 through the 16-channel FLUX-family VAE (generator environment).
usage: vae_apply.py <in.npy> <out.npy>"""
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL

src = np.load(sys.argv[1], mmap_mode="r")
vae = AutoencoderKL.from_pretrained("ostris/Flex.1-alpha", subfolder="vae", torch_dtype=torch.float32).cuda().eval()
out = np.lib.format.open_memmap(sys.argv[2], mode="w+", dtype=np.uint8, shape=src.shape)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for s in range(0, len(src), 8):
        x = torch.from_numpy(np.array(src[s:s + 8])).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
        out[s:s + 8] = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        if s % 640 == 0:
            print("VAE_APPLY", s, "/", len(src), flush=True)
out.flush()
print("VAE_APPLY_DONE", len(src), flush=True)
