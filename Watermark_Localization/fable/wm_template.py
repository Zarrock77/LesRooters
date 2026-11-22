"""Batched kit embedders and the message template they define.

template(x, m) = embed(x, m) - embed(x, 1 - m): the part of the watermark that depends on the message, as the
official embedder would write it on this very picture, in grey levels. The picture content cancels.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed(wm, x, bits):
    """x [B,3,H,W] in [0,1] on the embedder's device, bits [B,nbits] in {0,1} -> watermarked float picture."""
    if wm.name == "pixelseal":
        return wm.model.embed(x, msgs=bits.float(), is_video=False)["imgs_w"].clamp(0, 1)
    tm = wm.tm                                               # same steps as trustmark.encode, batched
    res = tm.model_resolution_enc
    cover = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    stego, _ = tm.encoder(cover, bits.float())
    residual = stego.clamp(-1, 1) - cover
    residual = residual - residual.mean(dim=(2, 3), keepdim=True)
    residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear")
    return ((x * 2 - 1 + residual).clamp(-1, 1) + 1) / 2


@torch.no_grad()
def template(wm, x, bits):
    return (embed(wm, x, bits) - embed(wm, x, 1 - bits)).mul(127.5)
