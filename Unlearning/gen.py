"""Unseen-like images for T3: SDXL-turbo img2img variations of the forget images (same scene, different instance).

No retain / held-out data is given, and crops or mixups of forget images still contain the memorised pixels. A variation
keeps the class and the look but is an image the original model never trained on, so orig's prediction on it stands in
for (a) the hidden control / validation images (retain anchor, gate monitor) and (b) what a model predicts on an unseen
image of that kind.

Usage: python gen.py <shard> <nshards> <out.npz> [strengths, comma-separated] [seed]      (needs diffusers + the sdxl-turbo snapshot in HF_HOME, offline)
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image

shard, nshard, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
strengths = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.4, 0.6, 0.8]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
imgs = np.load("forget_images.npz")["images"]
idx = np.arange(len(imgs))[shard::nshard]
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.vae.enable_slicing()  # decode one image at a time: a batch of 32 at 512px ran out of memory in the fp32 VAE
BS = 8
g = torch.Generator("cuda").manual_seed(shard + 1000 * seed)
res, src, st = [], [], []
for s in strengths:  # default 0.4 / 0.6 / 0.8; 4 steps * strength = 1, 2, 3 denoising steps: from near-copy to clearly different image
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        x = F.interpolate(torch.from_numpy(imgs[b]).permute(0, 3, 1, 2).float() / 255, size=512, mode="bicubic").clamp(0, 1)
        o = pipe(prompt=["a photo"] * len(b), image=x, strength=s, num_inference_steps=4, guidance_scale=0.0,
                 output_type="pt", generator=g).images
        o = F.interpolate(o.float(), size=224, mode="bicubic", antialias=True).clamp(0, 1)
        res.append((o * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        src.append(b)
        st.append(np.full(len(b), s, np.float32))
    print(f"shard {shard} strength {s} done", flush=True)
np.savez(out, images=np.concatenate(res), src=np.concatenate(src), strength=np.concatenate(st))
print("saved", out)
