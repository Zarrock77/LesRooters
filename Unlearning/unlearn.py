"""T3: fit the forget images to an emulated "gold" target, anchor everything else to orig.

forget term : KL(target || model(x)) on mildly augmented forget images. target = orig averaged over heavy crops (what orig
              predicts without the memorised detail), optionally shifted by the class prior of a model retrained without
              the forget set (--prior): classes that are entirely in the forget set can never be predicted by gold.
retain term : KL(orig || model) at T=2 on proxies for unseen data (none is given): mixup/cutmix of forget pairs,
              targeted-PGD images (--adv), heavy crops (--crop_r).
Only the last --blocks transformer blocks + norm + head are trained.

Usage: python unlearn.py --data . --strong 0.05 --tf 1 --sharpen 20 --adv 0.25 --prior 1 --out output/sub.pt
"""
import argparse
import copy
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=".")
ap.add_argument("--tf", type=float, default=3.0, help="temperature of the softened forget target")
ap.add_argument("--epochs", type=int, default=3, help="0 = only report (use with --init)")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--blocks", type=int, default=4, help="train only the last N blocks (+norm+head)")
ap.add_argument("--w_retain", type=float, default=1.0)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--strong", type=float, default=0.0,
                help=">0: forget target = mean softmax(orig/tf) over 16 strong augs (crop scale [strong, strong+0.3])")
ap.add_argument("--sharpen", type=float, default=1.0, help="target <- target**a renormalised (confident wrongs -> wider loss spread)")
ap.add_argument("--jit", type=float, default=0.0,
                help=">0: JiT forget loss (Foster et al. 2024), sigma = this value: ||f(x)-f(x+xi)|| / ||xi|| on logits")
ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
ap.add_argument("--adv", type=float, default=0.0,
                help=">0: retain proxy also uses targeted-PGD images (L_inf eps, in [-1,1] units) pushed to random classes (ZS-PAG-like)")
ap.add_argument("--prior", type=float, default=-1.0,
                help=">=0: shift the target by gold's class prior, p(j) *= (remaining_j / ncls)**prior, and mask classes "
                     "that are entirely in the forget set (gold never saw them, so it cannot predict them)")
ap.add_argument("--ncls", type=int, default=130, help="assumed training images per class (8 forget classes cap at exactly 130)")
ap.add_argument("--lf", type=float, default=0.0,
                help=">0: regress the true-class CE loss of images from fully forgotten classes to this value")
ap.add_argument("--dk", type=float, default=0.0,
                help=">0: among wrong classes also follow orig's logits on the full image (weight on the logit gap to the "
                     "best surviving wrong class); heavy crops alone vote for scene classes, not for the sibling object")
ap.add_argument("--sharpen_f", type=float, default=0.0,
                help=">0: sharpen factor for images of fully forgotten classes (their loss is set by --lf, so the target can stay hedged)")
ap.add_argument("--smix", action="store_true",
                help="retain proxy also mixes two forget images of the SAME class (class content without a memorised instance), "
                     "and all mixes use lambda in [0.3, 0.7] so that a retain image is never a near-copy of a forget image")
ap.add_argument("--crop_r", action="store_true", help="retain proxy also uses heavy crops (scale 0.05-0.35) of forget images")
ap.add_argument("--var", default="", help="glob of npz shards from gen.py: img2img variations of the forget images = unseen-like data")
ap.add_argument("--var_r", type=float, default=0.0, help=">0: retain proxy also uses the variations with strength >= this value")
ap.add_argument("--tgt_beta", type=float, default=0.0,
                help=">0: variation teacher. Target (at T=2) = mean softmax(orig/2) over the image's own unseen-like variations "
                     "(--vt_file, from vt_stats.py), with only the label coordinate sharpened: logit(p_label) *= beta")
ap.add_argument("--vt_file", default="/p/project1/training2666/delegue1/t3gen/vt.pt")
ap.add_argument("--vt", type=float, default=0.0,
                help=">0: forget target (also) = mean softmax(orig/tf) over the variations with strength >= this value")
ap.add_argument("--t2", action="store_true",
                help="fit the target at temperature 2 (use --tf 2): softmax(model/2) is exactly what the prediction score compares to gold")
ap.add_argument("--flip", type=float, default=-1.0,
                help=">=0: logit-surgery targets at T=2 (needs --strong). Images are ranked by how likely gold gets them wrong; "
                     "this top fraction is flipped: the alt class gets the label's logit, the label goes --m below it")
ap.add_argument("--m", type=float, default=8.0, help="logit margin of a flip = its loss (gold: loss mean + std ~ 6.5)")
ap.add_argument("--hedge", type=float, default=0.0,
                help="next fraction of the ranking: alt is raised to (label - hd), i.e. the mass is shared at T=2 instead of bet")
ap.add_argument("--hd", type=float, default=0.0)
ap.add_argument("--hd_by", default="", help="make the hedge depend on a ranking signal (crops|margin|var|depth|cap|avg)")
ap.add_argument("--hd_span", type=float, default=0.0,
                help="with --hd_by: hd goes from hd - span (image most likely wrong for gold: alt favoured) to hd + span")
ap.add_argument("--keep_drop", type=float, default=0.0, help="remaining images: label logit lowered by this much")
ap.add_argument("--rank", default="crops", choices=["crops", "margin", "var", "depth", "cap", "avg"],
                help="ranking signal: label mass under heavy crops / own logit margin / label prob on the s=0.6 variation / prediction depth / forget count of the class (capped classes first) / mean rank")
ap.add_argument("--topk", type=int, default=5, help="alt = class with most teacher mass among orig's top-k runner-ups on the full image")
ap.add_argument("--neg", action="store_true",
                help="retain proxy also uses the s=0.6 variation of each image of the batch: same scene, not the forget image, "
                     "so the change must stay on the instance instead of spilling to look-alikes")
ap.add_argument("--alt_var", action="store_true",
                help="tier mode: spread the wrong-class mass of flipped/hedged images like orig on the image's own variations "
                     "(--vt_file) instead of lifting orig's runner-up group; the label logit is placed as before")
ap.add_argument("--lift", type=int, default=1,
                help="raise orig's top-N runner-ups together (their mutual gaps kept) instead of the single alt: hedges over the siblings")
ap.add_argument("--v04", type=float, default=0.0, help="share of forget views replaced by the s=0.4 variation (a near-copy with altered texture)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--alt", type=float, default=0.0,
                help="share of forget views also corrupted like the hidden 'altered copies' (half of the forget score); "
                     "the target is built from orig on the corrupted view, exactly like for the clean views")
ap.add_argument("--alt_ops", default="blur,noise,jpeg,resize,color",
                help="corruptions drawn per image for --alt (report() always measures all five)")
ap.add_argument("--tail", type=float, default=0.0,
                help=">0 (with --init, --lf): only push the loss of this fraction of forget images (the ones --init already gets most "
                     "wrong) to --lf and keep everything else as --init: isolates the mean/std term of the score")
ap.add_argument("--lam", type=float, default=1.0,
                help="with --init: start from orig + lam * (init - orig). With --epochs 0 the result is reported and saved to --out: "
                     "drift on unseen data shrinks faster than the forget effect when pulling back toward orig")
ap.add_argument("--audit", type=float, default=0.0,
                help=">0: tail-mass audit with this floor, then exit (needs --init, --var = shards the model never loaded)")
ap.add_argument("--audit_ref", default="", help="with --audit: also paired KL(ref||sib) - KL(model||sib) against this checkpoint")
ap.add_argument("--tail_mass_min", type=float, default=0.0, help=">0: T=2 target mass outside label + 5 runner-ups raised to this (retail)")
ap.add_argument("--tail_unif", type=float, default=0.5, help="shape of the added tail mass: 0 = target's own tail, 1 = uniform")
ap.add_argument("--w_tail", type=float, default=0.0, help="binary KL on the mass outside label + 5 runner-ups (target vs student)")
ap.add_argument("--no_head", action="store_true", help="keep final norm + head frozen (only the blocks learn the edit)")
ap.add_argument("--only", default="", help="comma-separated parameter names: train only these")
ap.add_argument("--m2", default="", help="M2: 'mean,std' target T=1 CE moments; per-image CE L_i = smooth monotone function of the variant label mass, replaces the flip/hedge step")
ap.add_argument("--rank_var", default="", help="glob of extra variation shards used only for --rank var")
ap.add_argument("--rerank_head", action="store_true", help="permute the 5 lifted target logits by orig head geometry (codex_r6)")
ap.add_argument("--rank2", default="", choices=["", "crops", "margin", "depth"], help="keep --rank's flip count per class, choose the images inside each class by this signal (claude_r6)")
ap.add_argument("--lh", type=float, default=0.0, help=">0: regress the hedge-tier CE to this value")
ap.add_argument("--w_lf", type=float, default=1.0, help="weight of the flip-tier CE regression")
ap.add_argument("--repair", default="", help="checkpoint whose outputs on forget views are the forget target (drift repair pass)")
ap.add_argument("--w_ce", type=float, default=1.0, help="--repair: weight of the per-view CE match")
ap.add_argument("--ce_off", type=float, default=0.0, help="--repair: add this to the pinned CE (the pass loses ~0.1 of mean loss)")
ap.add_argument("--heal", type=int, default=0, help=">0: perturb the trained blocks (--reinit / --noise), then this many steps of distilling orig on unseen-like variations ONLY (no forget image, no handcrafted target), save, exit")
ap.add_argument("--reinit", type=int, default=0, help="--heal: re-initialise the last k blocks")
ap.add_argument("--noise", type=float, default=0.0, help="--heal: gaussian noise on the trained matrices, in units of each tensor's std")
ap.add_argument("--init", default="", help="start from this state dict instead of the original model")
ap.add_argument("--out", default="output/submission.pt")
args = ap.parse_args()
print(vars(args))

DATA = Path(args.data)
sys.path.insert(0, str(DATA))
import task_template as tt  # noqa: E402

torch.manual_seed(args.seed)
dev = tt.DEVICE
orig = tt.load_model(DATA / "model.pt")
model = tt.load_model(args.init) if args.init else copy.deepcopy(orig)
if args.lam != 1.0:
    with torch.no_grad():
        for p, q in zip(model.parameters(), orig.parameters()):
            p.copy_(q + args.lam * (p - q))
imgs = tt.load_forget_images(DATA / "forget_images.npz")
N = len(imgs)
lo = tt.predict(orig, imgs)
y = lo.argmax(1)  # true labels unknown: orig memorised the forget set (maxprob 0.9997), so its argmax stands in
remain = (args.ncls - torch.bincount(y, minlength=100)).clamp_min(0)  # training images gold still has, per class
cls_idx = [(y == c).nonzero().flatten() for c in range(100)]
is_f = (remain <= 1)[y]  # image belongs to a class that is (almost) entirely in the forget set
print(f"fully forgotten classes {(remain <= 1).nonzero().flatten().tolist()} = {int(is_f.sum())} images", flush=True)

vimg = None
if glob(args.var):
    shards = [np.load(f) for f in sorted(glob(args.var))]
    vimg = torch.from_numpy(np.concatenate([d["images"] for d in shards]))
    vsrc, vstr = (torch.from_numpy(np.concatenate([d[k] for d in shards])) for k in ("src", "strength"))
    vsel = torch.arange(0, len(vimg), 4)  # report subset
    vlo = tt.predict(orig, vimg[vsel])
    print(f"variations: {len(vimg)} images, strengths {vstr.unique().tolist()}", flush=True)

teach = orig
teachF = teach
if args.repair:  # second pass: forget views copy this checkpoint's own (server-validated) outputs, everything else -> orig
    assert args.flip < 0 and args.strong == 0 and args.t2, "--repair needs --flip -1 --strong 0 --t2 (--tf 2: KL at T=2)"
    teachF = tt.load_model(args.repair)
if args.tail > 0:
    teach = copy.deepcopy(model)  # everything except the tail stays as the init model
    L0 = F.cross_entropy(tt.predict(model, imgs), y, reduction="none")
    is_f = L0 >= L0.kthvalue(int(N * (1 - args.tail))).values
    print(f"tail: {int(is_f.sum())} images, init loss {L0[is_f].mean():.2f}+-{L0[is_f].std():.2f} -> {args.lf}", flush=True)

for p in model.parameters():
    p.requires_grad = False
nb = len(model.blocks)
train_mods = [*model.blocks[nb - args.blocks:]] + ([] if args.no_head else [model.norm, model.head])  # frozen readout: no class-wide head shortcut
params = [p for m in train_mods for p in m.parameters()]
if args.only:  # e.g. blocks.11.mlp.fc2.weight: a one-matrix edit
    params = [p for n, p in model.named_parameters() if n in args.only.split(",")]
    assert params, f"--only matched nothing: {args.only}"
for p in params:
    p.requires_grad = True
print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")


def aug(x, smin=0.6, smax=1.0, rot=0.0):
    """x float [B,3,224,224] preprocessed. Random hflip + random resized crop (scale smin-smax) + small brightness."""
    B = x.shape[0]
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    s = torch.empty(B, device=dev).uniform_(smin, smax).sqrt()
    tx, ty = (torch.rand(2, B, device=dev) * 2 - 1) * (1 - s)
    a = torch.empty(B, device=dev).uniform_(-rot, rot) * 3.14159 / 180
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = s * a.cos(), -s * a.sin(), s * a.sin(), s * a.cos()
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    return (x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)).clamp(-1, 1)


ALT_OPS = ("blur", "noise", "jpeg", "resize", "color")
try:
    from torchvision.io import decode_jpeg, encode_jpeg
except ImportError:  # no torchvision: 'jpeg' falls back to a coarse down/up-sample
    encode_jpeg = None


def corrupt(x, op, g=None):
    """x float [B,3,224,224] in [-1,1]. One corruption family, random strength per image (moderate: the copy keeps its content)."""
    B = x.shape[0]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=x.device, generator=g)  # noqa: E731
    if op == "blur":  # separable gaussian, sigma 0.6-1.8 px, per image
        sig, r = u(0.6, 1.8).view(B, 1, 1), 4
        k = torch.exp(-(torch.arange(-r, r + 1, device=x.device).float().view(1, 1, -1) ** 2) / (2 * sig ** 2))
        k = (k / k.sum(-1, keepdim=True)).repeat_interleave(3, 0)  # [3B,1,9]
        y = F.pad(x.reshape(1, 3 * B, 224, 224), (r, r, r, r), mode="reflect")
        y = F.conv2d(y, k.view(3 * B, 1, 1, -1), groups=3 * B)
        y = F.conv2d(y, k.view(3 * B, 1, -1, 1), groups=3 * B)
        return y.view(B, 3, 224, 224).clamp(-1, 1)
    if op == "noise":  # gaussian, sigma 0.02-0.10 in [-1,1] units (~2.5-13 /255)
        n = torch.randn(x.shape, device=x.device, generator=g)
        return (x + u(0.02, 0.10) * n).clamp(-1, 1)
    if op == "resize":  # down to 40-75 % then back up
        out = torch.empty_like(x)
        for i, s in enumerate(u(0.4, 0.75).flatten().tolist()):
            small = F.interpolate(x[i:i + 1], size=int(224 * s), mode="bilinear", align_corners=False, antialias=True)
            out[i:i + 1] = F.interpolate(small, size=224, mode="bilinear", align_corners=False)
        return out.clamp(-1, 1)
    if op == "color":  # saturation x0.5-1.5, contrast x0.7-1.3
        gray = x.mean(1, keepdim=True)
        x = gray + (x - gray) * u(0.5, 1.5)
        m = x.mean((1, 2, 3), keepdim=True)
        return (m + (x - m) * u(0.7, 1.3)).clamp(-1, 1)
    if op == "jpeg":  # quality 25-75, done on CPU uint8
        if encode_jpeg is None:
            return corrupt(corrupt(x, "resize", g), "noise", g)
        q = torch.randint(25, 76, (B,), generator=None).tolist()
        u8 = ((x.float().clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8).cpu()
        dec = torch.stack([decode_jpeg(encode_jpeg(u8[i], quality=q[i])) for i in range(B)])
        return (dec.to(x.device).float() / 255 - 0.5) / 0.5
    raise ValueError(op)


alt_ops = [o.strip() for o in args.alt_ops.split(",") if o.strip()]
assert alt_ops and all(o in ALT_OPS for o in alt_ops), f"--alt_ops must be a subset of {ALT_OPS}"


def corrupt_mixed(x, ops):
    """Each image gets one corruption drawn from ops."""
    pick = torch.randint(len(ops), (x.shape[0],), device=x.device)
    out = x.clone()
    for j, op in enumerate(ops):
        m = pick == j
        if m.any():
            out[m] = corrupt(x[m], op).to(out.dtype)
    return out


def mix(x, xp=None, lo=0.0, hi=1.0):
    """Mixup or cutmix with a partner batch (default: x shuffled): images the model never saw as such."""
    lam = lo + (hi - lo) * torch.rand(1).item()
    if xp is None:
        xp = x[torch.randperm(x.shape[0], device=dev)]
    if torch.rand(1).item() < 0.5:
        return lam * x + (1 - lam) * xp
    h = int(224 * (1 - lam) ** 0.5)
    i, j = torch.randint(0, 224 - h + 1, (2,)).tolist()
    x = x.clone()
    x[:, :, i:i + h, j:j + h] = xp[:, :, i:i + h, j:j + h]
    return x


def same_class(idx):
    """For each forget index, a random forget image of the same class (preprocessed)."""
    return tt.preprocess(imgs[torch.stack([cls_idx[c][torch.randint(len(cls_idx[c]), (1,))] for c in y[idx].tolist()]).flatten()])


def kl(p_logits, q_logits, T):
    """KL(softmax(p/T) || softmax(q/T)), batch mean."""
    return F.kl_div(F.log_softmax(q_logits / T, 1), F.log_softmax(p_logits / T, 1), log_target=True, reduction="batchmean")


def both(x):
    """orig and model logits on preprocessed images x (cpu), server-style fp16."""
    with torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
        return [torch.cat([m(x[i:i + 128].to(dev)).float() for i in range(0, len(x), 128)]) for m in (orig, model)]


target = tier = None


@torch.no_grad()
def report(tag):
    ls = tt.predict(model, imgs)
    L = F.cross_entropy(ls, y, reduction="none")
    ok = ls.argmax(1) == y
    sub = [tt.preprocess(imgs[i:i + 128]) for i in range(0, 1024, 128)]
    mo, ms = both(torch.cat([mix(x).cpu() for x in sub]))            # unseen-data proxy 1: mixup / cutmix
    co, cs = both(torch.cat([aug(x, 0.05, 0.35).cpu() for x in sub]))  # proxy 2: heavy crops (class-level damage)
    so, ss = both(torch.cat([mix(x, same_class(torch.arange(i * 128, i * 128 + len(x))), 0.3, 0.7).cpu() for i, x in enumerate(sub)]))
    _, al = both(torch.cat([(aug(x, 0.8, 1.0, 8) + 0.05 * torch.randn_like(x)).clamp(-1, 1).cpu() for x in sub]))
    La = F.cross_entropy(al.cpu(), y[:1024], reduction="none")       # stand-in for the hidden altered copies
    alt_line = ""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == "cuda" else []):
        torch.manual_seed(0)  # same corrupted copies at every report and in every run
        for op in ALT_OPS:  # each corruption family alone: does the forget behaviour carry over to altered copies?
            ao, asub = both(torch.cat([corrupt(x, op).cpu() for x in sub]))
            Lo = F.cross_entropy(ao.cpu(), y[:1024], reduction="none")
            Ls = F.cross_entropy(asub.cpu(), y[:1024], reduction="none")
            alt_line += (f" {op}: orig {Lo.mean():.2f}+-{Lo.std():.2f} sub {Ls.mean():.2f}+-{Ls.std():.2f}"
                         f" agree {(ao.argmax(1) == asub.argmax(1)).float().mean():.2f} |")
    fm = is_f[:1024]

    def cacc(t, m):  # does the class concept survive on views that are not near-copies? (gate / retain-accuracy proxy)
        return (t.argmax(1).cpu() == y[:1024])[m].float().mean()

    print(f"[{tag}] forget: loss mean {L.mean():.3f} std {L.std():.3f} sum {L.mean() + L.std():.2f}"
          f"  agree-w-orig {ok.float().mean():.3f}  maxprob {ls.softmax(1).max(1).values.mean():.3f}"
          f"  KL(orig||sub)T2 {kl(lo, ls, 2).item():.3f}"
          f" | F: loss {L[is_f].mean():.2f}+-{L[is_f].std():.2f}  P: loss {L[~is_f].mean():.2f}+-{L[~is_f].std():.2f} acc {ok[~is_f].float().mean():.3f}"
          + (f" tgt-agree {(ls.argmax(1) == target.argmax(1).cpu()).float().mean():.3f}"
             f" tgt-KL {F.kl_div(F.log_softmax(ls / (2 if args.t2 else 1), 1), target.cpu(), reduction='batchmean').item():.3f}"
             if target is not None else "")
          + f" | altered: loss mean {La.mean():.3f} std {La.std():.3f}"
          f" | mix: KL(orig||sub)T2 {kl(mo, ms, 2).item():.4f} agree {(mo.argmax(1) == ms.argmax(1)).float().mean():.3f}"
          f" | same-class mix: KL {kl(so, ss, 2).item():.4f} label-acc F/P orig {cacc(so, fm):.3f}/{cacc(so, ~fm):.3f} sub {cacc(ss, fm):.3f}/{cacc(ss, ~fm):.3f}"
          f" | crop: KL(orig||sub)T2 {kl(co, cs, 2).item():.4f} agree {(co.argmax(1) == cs.argmax(1)).float().mean():.3f}"
          f" label-acc F/P orig {cacc(co, fm):.3f}/{cacc(co, ~fm):.3f} sub {cacc(cs, fm):.3f}/{cacc(cs, ~fm):.3f}",
          flush=True)
    print("   altered copies (first 1024, loss vs orig-argmax label; clean forget-set loss is the line above):" + alt_line,
          flush=True)
    if tier is not None:
        p2, ar = (ls / 2).softmax(1), torch.arange(N)
        print("   " + " | ".join(f"{name} {int((tier == k).sum())}: loss {L[tier == k].mean():.2f}+-{L[tier == k].std():.2f}"
                                 f" T2 p(label) {p2[ar, y][tier == k].mean():.3f} p(alt) {p2[ar, alt][tier == k].mean():.3f}"
                                 for k, name in ((2, "flip"), (1, "hedge"), (0, "keep")) if (tier == k).any()), flush=True)
    if vimg is not None:  # unseen-like images: does sub still behave like orig there? (what the retain score and the gate measure)
        vs, yv, fv = tt.predict(model, vimg[vsel]), y[vsrc[vsel]], is_f[vsrc[vsel]]
        for st in vstr.unique().tolist():
            line = f"   var s={st:.1f}:"
            for name, m in (("F", fv), ("P", ~fv)):
                m = m & (vstr[vsel] == st)
                line += (f" {name} acc orig {(vlo[m].argmax(1) == yv[m]).float().mean():.3f} sub {(vs[m].argmax(1) == yv[m]).float().mean():.3f}"
                         f" agree {(vlo[m].argmax(1) == vs[m].argmax(1)).float().mean():.3f} KL {kl(vlo[m], vs[m], 2).item():.3f}"
                         f" maxprob orig {vlo[m].softmax(1).max(1).values.mean():.3f} |")
            print(line, flush=True)
        far, conf = vstr[vsel] > 0.55, vlo.softmax(1).max(1).values > 0.9  # typical control image = orig is confident
        print("   var >=0.6, orig confident / not: " + " / ".join(
            f"KL {kl(vlo[m], vs[m], 2).item():.3f} agree {(vlo[m].argmax(1) == vs[m].argmax(1)).float().mean():.3f} (n {int(m.sum())})"
            for m in (far & conf, far & ~conf)), flush=True)


def retail(p, Gi, tmin, unif):
    """p [B,100] T=2 probs. Raise the mass outside Gi (label + 5 runner-ups) to >= tmin, ratios inside Gi kept; the added
    mass follows p's own tail (unif=0) or is spread evenly over the 94 other classes (unif=1). Rows above tmin: unchanged."""
    inG = torch.zeros_like(p, dtype=torch.bool).scatter_(1, Gi, True)
    s = (p * ~inG).sum(1, keepdim=True)
    add = (tmin - s).clamp_min(0)
    shape = (1 - unif) * (p * ~inG) / s.clamp_min(1e-12) + unif * (~inG).float() / 94
    return torch.where(inG, p * (1 - s - add) / (1 - s), p * ~inG + add * shape)


@torch.no_grad()
def audit(tmin):
    """Step 0 of output/debate/t3/plan_final.md: student mass outside G_i, sibling-reference mass, and the paired KL saving
    of retail() applied to the student's own outputs (no training). --var must hold only shards the model never loaded."""
    G = lo.topk(6, 1).indices  # label + orig's 5 clean runner-ups, fixed per source
    sib = (vstr > 0.45) & (vstr < 0.65)
    R = (tt.predict(orig, vimg[sib]) / 2).softmax(1).double()
    js = vsrc[sib].long()
    inGs = torch.zeros_like(R, dtype=torch.bool).scatter_(1, G[js], True)
    g = (R * ~inGs).sum(1)
    tail = (R * ~inGs).sort(1, descending=True).values
    print(f"[audit] siblings {len(R)} on {len(js.unique())} sources | reference g outside G: mean {g.mean():.4f} median "
          f"{g.median():.4f} | share of g in top-1 tail class {(tail[:, 0] / g.clamp_min(1e-12)).mean():.3f}, top-5 "
          f"{(tail[:, :5].sum(1) / g.clamp_min(1e-12)).mean():.3f}", flush=True)
    xs = [tt.preprocess(imgs[i:i + 128]) for i in range(0, N, 128)]
    ref = tt.load_model(args.audit_ref) if args.audit_ref else None

    def views_of(m):
        v = {"clean": tt.predict(m, imgs)}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == "cuda" else []):
            torch.manual_seed(0)  # identical corrupted copies for model and ref
            for op in ALT_OPS:
                cx = torch.cat([corrupt(x, op).cpu() for x in xs])
                with torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):  # server-style, like both()
                    v[op] = torch.cat([m(cx[i:i + 128].to(dev)).float().cpu() for i in range(0, N, 128)])
        return v

    views = views_of(model)
    rviews = views_of(ref) if ref is not None else None
    gen = torch.Generator().manual_seed(0)
    for vname, ls in views.items():
        P = (ls / 2).softmax(1).double()
        s = 1 - P.gather(1, G).sum(1)
        line = f"[audit {vname}] student s outside G: mean {s.mean():.4f} median {s.median():.4f} >={tmin}: {(s >= tmin).float().mean():.3f}"
        lp = P[js].clamp_min(1e-300).log()
        base = (R * (R.clamp_min(1e-300).log() - lp)).sum(1)
        for unif in (0.0, 0.5, 1.0):
            sav = (R * (retail(P, G, tmin, unif)[js].clamp_min(1e-300).log() - lp)).sum(1)  # KL(R||P) - KL(R||Q)
            per = torch.zeros(N, dtype=torch.float64).index_add_(0, js, sav) / torch.bincount(js, minlength=N).clamp_min(1)
            for name, par in (("sel", 0), ("dec", 1)):
                src = js.unique()
                src = src[src % 2 == par]
                v = per[src]
                bs = v[torch.randint(len(v), (1000, len(v)), generator=gen)].mean(1)
                line += (f" | {name} unif={unif}: saving {v.mean():.4f} [{bs.quantile(0.025):.4f}, {bs.quantile(0.975):.4f}]"
                         f" of KL {base[(js % 2) == par].mean():.3f}")
        print(line, flush=True)
        if rviews is not None:  # local comparison of plan_final.md: per-source paired KL saving of model over ref
            lr_ = (rviews[vname] / 2).log_softmax(1).double()[js]
            d = (R * (lp - lr_)).sum(1)  # KL(R||ref) - KL(R||model)
            per = torch.zeros(N, dtype=torch.float64).index_add_(0, js, d) / torch.bincount(js, minlength=N).clamp_min(1)
            for name, par in (("sel", 0), ("dec", 1)):
                src = js.unique()
                v = per[src[src % 2 == par]]
                bs = v[torch.randint(len(v), (1000, len(v)), generator=gen)].mean(1)
                line2 = f"[vs-ref {vname} {name}] KL saving over ref {v.mean():.4f} [{bs.quantile(0.025):.4f}, {bs.quantile(0.975):.4f}]"
                print(line2, flush=True)


if args.audit > 0:
    audit(args.audit)
    sys.exit()

if args.heal > 0:
    # The last blocks hold the instance memory (CLS kNN accuracy 54 % at block 10 = gold-like, 99.6 % at block 12). Wipe it there,
    # then relearn orig's function from images orig never trained on: what the result predicts on a forget image is what a
    # model with orig's knowledge but without that image predicts, per image and from its own pixels.
    with torch.no_grad():
        for blk in (model.blocks[nb - args.reinit:] if args.reinit else []):
            for n_, p in blk.named_parameters():
                if n_.endswith("gamma"):
                    continue  # keep the trained residual scale
                if "norm" in n_:
                    p.fill_(1.0 if n_.endswith("weight") else 0.0)
                elif n_.endswith("weight"):
                    torch.nn.init.trunc_normal_(p, std=0.02)
                else:
                    p.zero_()
        if args.noise > 0:
            for p in params:
                if p.dim() > 1:
                    p.add_(args.noise * p.std() * torch.randn_like(p))
    keep = vstr > 0.45
    keep[vsel] = False  # report() measures drift on vsel: never trained on
    hp = keep.nonzero().flatten()
    print(f"heal: {len(hp)} unseen-like variations, reinit {args.reinit}, noise {args.noise}, {args.heal} steps", flush=True)
    report("perturbed")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.heal, pct_start=0.1)
    model.train()
    for k in range(args.heal):
        xv = aug(tt.preprocess(vimg[hp[torch.randint(len(hp), (96,))]]))
        xr = torch.cat([xv[:48], mix(xv[48:72], None, 0.3, 0.7), aug(xv[72:], 0.2, 0.6)])
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            tr = orig(xr).float()
        with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            sr = model(xr).float()
        loss = kl(tr, sr, 1.0) + 4 * kl(tr, sr, 2.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if k % 100 == 0:
            print(f"heal it {k} loss {loss.item():.4f}", flush=True)
        if (k + 1) % (args.heal // 3) == 0:
            model.eval()
            report(f"heal{k + 1}")
            model.train()
    model.eval()
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values())
    tt.build_model().load_state_dict(sd, strict=True)
    torch.save(sd, args.out)
    torch.save({"logits": tt.predict(model, imgs)}, args.out.replace(".pt", "_fl.pt"))
    print("saved", args.out)
    sys.exit()

report("start")
if args.epochs == 0:
    if args.lam != 1.0:
        torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, args.out)
        print("saved", args.out)
    sys.exit()

if args.tgt_beta > 0:
    # orig on variations of the image reproduces gold's loss stats (CE 2.49 +- 3.91 vs gold mean+std ~6.5); averaging over
    # 5 variations makes it too soft, so the label coordinate alone is sharpened and the wrong-class mass keeps its shape
    q, ar = torch.load(args.vt_file)["q"], torch.arange(N)
    py = torch.sigmoid(args.tgt_beta * torch.logit(q[ar, y].clamp(1e-4, 1 - 1e-4)))
    rest = q.clone()
    rest[ar, y] = 0
    target = rest / rest.sum(1, keepdim=True) * (1 - py)[:, None]
    target[ar, y] = py
    assert torch.allclose(target.sum(1), torch.ones(N), atol=1e-4)
    Lt = F.cross_entropy(2 * target.clamp_min(1e-12).log(), y, reduction="none")
    print(f"variation target: implied loss {Lt.mean():.2f} +- {Lt.std():.2f}, p(label) T2 {py.mean():.3f},"
          f" label argmax {(target.argmax(1) == y).float().mean():.3f}", flush=True)
    args.t2, target = True, target.to(dev)
elif args.strong > 0 or args.vt > 0:
    parts = []
    if args.strong > 0:
        # orig on heavy crops = what a model without memorised fine detail predicts; ambiguous images go to plausible wrong classes
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
            t = torch.zeros(N, 100)
            for i in range(0, N, 128):
                x = tt.preprocess(imgs[i:i + 128])
                for _ in range(16):
                    t[i:i + 128] += F.softmax(orig(aug(x, args.strong, min(1.0, args.strong + 0.3))).float() / args.tf, 1).cpu()
        parts.append(t / 16)
    if args.vt > 0:
        # orig on unseen-like variations of the same image: the closest thing to "a model that never saw this image"
        m = vstr >= args.vt
        p = F.softmax(tt.predict(orig, vimg[m]) / args.tf, 1)
        parts.append(torch.zeros(N, 100).index_add_(0, vsrc[m], p) / torch.zeros(N).index_add_(0, vsrc[m], torch.ones(int(m.sum())))[:, None])
    target = sum(parts) / len(parts)
    if args.flip >= 0:
        pm, ar = target, torch.arange(N)  # teacher mass per class
        z = lo.clone()
        z[ar, y] = -1e9
        sc = {"crops": 1 - pm[ar, y], "margin": z.max(1).values - lo[ar, y]}  # higher = gold more likely wrong
        if vimg is not None:
            m6 = (vstr > 0.45) & (vstr < 0.65)  # unseen-like variations (0.5 / 0.6), several per image when more were generated
            pv = F.softmax(tt.predict(orig, vimg[m6]) / 2, 1)[torch.arange(int(m6.sum())), y[vsrc[m6]]]
            vs6 = vsrc[m6]
            if args.rank_var:  # extra variation shards used for this ranking only (less noisy per-image hardness)
                ex = [np.load(f) for f in sorted(glob(args.rank_var))]
                xi, xs, xt = (torch.from_numpy(np.concatenate([d[k] for d in ex])) for k in ("images", "src", "strength"))
                mx = (xt > 0.45) & (xt < 0.65)
                pv = torch.cat([pv, F.softmax(tt.predict(orig, xi[mx]) / 2, 1)[torch.arange(int(mx.sum())), y[xs[mx].long()]]])
                vs6 = torch.cat([vs6, xs[mx].long()])
                del xi
                print(f"rank_var: {len(vs6)} variations, {torch.bincount(vs6, minlength=N).float().mean():.1f} per image", flush=True)
            sc["var"] = 1 - torch.zeros(N).index_add_(0, vs6, pv) / torch.zeros(N).index_add_(0, vs6, torch.ones(len(vs6)))
        # prediction depth (Baldock et al. 2021): is the class already readable from earlier blocks, by leave-one-out kNN
        # among the forget images? Late-resolved images are the ones a model that never saw them tends to get wrong.
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
            cls = [[] for _ in orig.blocks]
            for i in range(0, N, 128):
                h = orig.norm_pre(orig._pos_embed(orig.patch_embed(tt.preprocess(imgs[i:i + 128]))))
                for b, blk in enumerate(orig.blocks):
                    h = blk(h)
                    cls[b].append(F.normalize(h[:, 0].float(), dim=1))
        vote, yd_ = {}, y.to(dev)
        for b in range(6, 12):
            f = torch.cat(cls[b])
            nn_ = (f @ f.T).fill_diagonal_(-2).topk(10, 1).indices
            vote[b] = (yd_[nn_] == yd_[:, None]).float().mean(1).cpu()
            if b == 9:  # neighbours' other classes at block 10: where an unmemorised reading of the image points
                kalt = torch.stack([torch.bincount(r[r != c], minlength=100) for r, c in zip(yd_[nn_].cpu(), y)])
        print("leave-one-out kNN: label share among 10 neighbours, blocks 7-12: " + " ".join(f"{vote[b].mean():.2f}" for b in range(6, 12)), flush=True)
        sc["depth"] = 1 - sum(vote[b] for b in range(7, 11)) / 4
        rk = {k: v.argsort().argsort().float() / N for k, v in sc.items()}
        ks = list(rk)
        print("rank correlation between signals: " + " ".join(
            f"{a}/{b} {torch.corrcoef(torch.stack([rk[a], rk[b]]))[0, 1]:.2f}" for i, a in enumerate(ks) for b in ks[i + 1:]), flush=True)
        avg = sum(rk.values()) / len(rk)
        rk["cap"] = torch.bincount(y, minlength=100)[y].float() + avg  # classes with the most forget images first
        order = (avg if args.rank == "avg" else rk[args.rank]).argsort(descending=True)
        nf, nh = int(args.flip * N), int(args.hedge * N)
        tier = torch.zeros(N, dtype=torch.long)  # 2 = flip, 1 = hedge, 0 = keep
        tier[order[:nf]] = 2
        tier[order[nf:nf + nh]] = 1
        if args.rank2:
            old = tier == 2
            for i in cls_idx:
                k = int(old[i].sum())
                if k:
                    tier[i] = tier[i].clamp_max(1)  # the class's flips go back to the hedge tier
                    tier[i[rk[args.rank2][i].argsort(descending=True)[:k]]] = 2  # same count, hardest by rank2
            assert torch.equal(torch.bincount(y[tier == 2], minlength=100), torch.bincount(y[old], minlength=100))
            print(f"rank2 {args.rank2}: {int((old & (tier == 2)).sum())} of {int(old.sum())} flips unchanged; "
                  f"distinct depth values per class (median) {torch.tensor([len(rk[args.rank2][i].unique()) / max(len(i), 1) for i in cls_idx if len(i)]).median():.2f}", flush=True)
        top = lo.topk(args.topk + 1, 1).indices[:, 1:]  # orig's runner-ups on the full image (the label is always top-1)
        alt = top.gather(1, pm.gather(1, top).argmax(1, keepdim=True)).squeeze(1)
        print(f"tiers: flip {nf} (m {args.m}) hedge {nh} (hd {args.hd}) keep {N - nf - nh} (drop {args.keep_drop});"
              f" alt = runner-up for {(alt == top[:, 0]).float().mean():.2f}; teacher label mass flip/hedge/keep "
              + "/".join(f"{pm[ar, y][tier == k].mean():.2f}" for k in (2, 1, 0)), flush=True)
        is_f, target, args.lf = tier == 2, None, args.m
        if args.lift > 1:  # runner-up group: alt first, then orig's next runner-ups
            grp = torch.cat([alt[:, None], torch.stack([r[r != a][:args.lift - 1] for r, a in zip(top, alt)])], 1)
        else:
            grp = alt[:, None]
        hdv = args.hd + args.hd_span * (1 - 2 * (avg if args.hd_by == "avg" else rk[args.hd_by]).argsort().argsort().float() / N) \
            if args.hd_by else torch.full((N,), args.hd)
        tierd, altd, grpd, hdd = tier.to(dev), alt.to(dev), grp.to(dev), hdv.to(dev)
        Gd = lo.topk(6, 1).indices.to(dev)  # label + orig's 5 clean runner-ups (= label + grp at --topk 5 --lift 5)
        if args.m2:
            # M2 of output/debate/t3/plan_final.md: a_i = sigmoid(al + be*logit(u_i)) with u_i = orig's T=2 label mass on the
            # image's unseen-like variations; (al, be) solved so the implied T=1 CE has the target moments. Label mass a at T=2
            # with off-label conditional r gives T=1 CE log(1 + S((1-a)/a)^2), S = sum r^2 (T=1 probs ~ T=2 probs squared).
            mu, sd = (float(v) for v in args.m2.split(","))
            u = (1 - sc["var"]).clamp(1e-3, 1 - 1e-3).double()
            ar = torch.arange(N)
            z = lo.clone().double()
            z.scatter_(1, grp, lo.gather(1, grp).double() + (lo[ar, y].double() - lo.gather(1, grp).double().max(1).values)[:, None])
            z[ar, y] = -1e9
            S = (z / 2).softmax(1).pow(2).sum(1)
            ab = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
            o2 = torch.optim.Adam([ab], lr=0.02)

            def m2_L(ab):
                a = torch.sigmoid(ab[0] + ab[1] * torch.logit(u))
                return torch.log1p(S * ((1 - a) / a) ** 2).clamp(max=12.0)
            for _ in range(4000):
                L_ = m2_L(ab)
                e = (L_.mean() - mu) ** 2 + (L_.std() - sd) ** 2
                o2.zero_grad()
                e.backward()
                o2.step()
            with torch.no_grad():
                L_ = m2_L(ab).float()
            assert ab[1].item() > 0 and abs(L_.mean() - mu) < 0.05 and abs(L_.std() - sd) < 0.05, (ab, L_.mean(), L_.std())
            Ld = L_.to(dev)
            args.lf = 0.0  # the per-image CE regression below replaces the flip-tier one
            print(f"m2: al {ab[0].item():.3f} be {ab[1].item():.3f} target CE {L_.mean():.3f}+-{L_.std():.3f}, at bound 12 "
                  f"{(L_ >= 11.99).float().mean():.3f}; mean target CE flip/hedge tiers " + "/".join(
                      f"{L_[tier == k].mean():.2f}" for k in (2, 1)), flush=True)
        if args.alt_var:  # T=2 logit offsets reproducing the variations' wrong-class ratios, best wrong class at 0
            q = torch.load(args.vt_file)["q"].clone()
            q[ar, y] = 0
            Rv = (2 * (q / q.max(1, keepdim=True).values).clamp_min(1e-30).log()).clamp_min(-40)
            Rv[ar, y] = 0
            print(f"alt_var: variation best wrong class = orig runner-up for {(q.argmax(1) == top[:, 0]).float().mean():.2f};"
                  f" wrong classes within 4 T2-logits of the best {(Rv > -4).sum(1).float().mean() - 1:.1f}", flush=True)
            Rd = Rv.to(dev)
    else:
        logt = target.clamp_min(1e-30).log()
        gone, ar = remain <= 1, torch.arange(N)
        if args.dk > 0:
            z = lo.clone()
            z[:, gone] = -1e9
            z[ar, y] = -1e9
            gap = lo - z.max(1, keepdim=True).values  # 0 for the best surviving wrong class, negative for the others
            gap[ar, y] = 0                            # the true class is judged by the crops alone
            srt = gap.masked_fill(z < -1e8, -1e9).sort(1, descending=True).values
            print(f"dark knowledge: logit gap of 2nd/3rd/5th/10th surviving wrong class {srt[:, 1].mean():.2f} {srt[:, 2].mean():.2f}"
                  f" {srt[:, 4].mean():.2f} {srt[:, 9].mean():.2f}", flush=True)
            logt = logt + args.dk * gap
        if args.prior >= 0:
            logw = (remain.float() / args.ncls).clamp_min(1e-30).log() * args.prior
            logw[remain <= 1] = -1e4
            logt = logt + logw
        # in log space: target**50 underflows to 0/0 = NaN in float32
        sh = torch.full((N, 1), args.sharpen)
        if args.sharpen_f > 0:
            sh[is_f] = args.sharpen_f
        target = F.softmax(sh * logt, 1)
        ty = target.argmax(1)
        print(f"target: agree-w-orig {(ty == y).float().mean():.3f} (P only {(ty == y)[~is_f].float().mean():.3f})"
              f"  maxprob {target.max(1).values.mean():.3f} (F {target.max(1).values[is_f].mean():.3f})", flush=True)
        for c in gone.nonzero().flatten().tolist():  # which wrong class does the target pick for a forgotten class?
            top = torch.bincount(ty[y == c], minlength=100).topk(4)
            print(f"  forgotten class {c} -> " + " ".join(f"{int(i)}:{int(v)}" for i, v in zip(top.indices, top.values) if v > 0), flush=True)
        target = target.to(dev)

pool = None
if args.adv > 0:
    # proxy for the hidden control set: forget images pushed (targeted PGD on orig) toward a random class out of all 100,
    # so the retain anchor covers the 55 classes absent from the forget set
    pool = torch.empty(N, 3, 224, 224, dtype=torch.float16)
    tcls = torch.randint(0, 100, (N,))
    for i in range(0, N, 128):
        x0 = tt.preprocess(imgs[i:i + 128])
        x, yt = x0.clone(), tcls[i:i + 128].to(dev)
        for _ in range(10):
            x.requires_grad_(True)
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                g, = torch.autograd.grad(F.cross_entropy(orig(x).float(), yt), x)
            x = (x.detach() - args.adv / 4 * g.sign()).clamp(x0 - args.adv, x0 + args.adv).clamp(-1, 1)
        pool[i:i + 128] = x.half().cpu()
    with torch.no_grad():
        hit = (tt.predict(orig, ((pool[:512].float() * 0.5 + 0.5) * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1)).argmax(1)
               == tcls[:512]).float().mean()
    print(f"adv pool: eps {args.adv}, orig predicts target class on {hit:.3f} of 512 checked", flush=True)

opt = (torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0) if args.opt == "adamw"
       else torch.optim.SGD(params, lr=args.lr, momentum=0.9))
steps = args.epochs * (N // args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
yd, fd = y.to(dev), is_f.to(dev)
if args.var_r > 0:  # retain anchors: variations far enough from the source, minus the ones the report uses
    keep = vstr >= args.var_r
    keep[vsel] = False
    vpool = keep.nonzero().flatten()
if args.neg:  # each image's s=0.6 variation, -1 when it is held out for the report
    m6 = (vstr > 0.45) & (vstr < 0.65)
    m6[vsel] = False
    v06 = torch.full((N,), -1, dtype=torch.long)
    v06[vsrc[m6]] = m6.nonzero().flatten()
if args.v04 > 0:
    v04 = torch.zeros(N, dtype=torch.long)
    v04[vsrc[vstr < 0.45]] = (vstr < 0.45).nonzero().flatten()
model.train()
for ep in range(args.epochs):
    perm = torch.randperm(N)
    for k in range(N // args.bs):
        idx = perm[k * args.bs:(k + 1) * args.bs]
        x = tt.preprocess(imgs[idx])
        xs = x
        if args.v04 > 0:
            use = torch.rand(len(idx)) < args.v04
            xs = x.clone()
            xs[use.to(dev)] = tt.preprocess(vimg[v04[idx[use]]])
        xf = aug(xs)
        if args.alt > 0:  # altered copies: same target recipe, computed by teach() on the corrupted view below
            use = torch.rand(len(idx), device=dev) < args.alt
            if use.any():
                xf = xf.clone()
                xf[use] = corrupt_mixed(xf[use], alt_ops)
        # retain batch: one share per proxy for unseen data
        lohi = (0.3, 0.7) if args.smix else (0.0, 1.0)
        src = [lambda n: mix(aug(x), None, *lohi)[:n]]
        if args.smix:
            src.append(lambda n: mix(aug(x[:n]), aug(same_class(idx[:n])), *lohi))
        if pool is not None:
            src.append(lambda n: aug(pool[torch.randint(0, N, (n,))].to(dev).float()))
        if args.crop_r:
            src.append(lambda n: aug(x[:n], 0.05, 0.35))
        if args.var_r > 0:
            src.append(lambda n: aug(tt.preprocess(vimg[vpool[torch.randint(len(vpool), (n,))]])))
        if args.neg:
            src.append(lambda n: aug(tt.preprocess(vimg[v06[idx][v06[idx] >= 0][:n]])))
        n = max(args.bs // 2, args.bs // len(src))
        xr = torch.cat([f(n) for f in src])
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            tf, tr = teachF(xf).float(), teach(xr).float()
        with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            sf, sr = model(xf).float(), model(xr).float()
        # forget target is orig softened by tf; compare at T=1 so the student actually becomes less confident
        if args.jit > 0:
            # JiT: local Lipschitz penalty around each forget point (one noise draw per step = stochastic estimate)
            xi = torch.randn_like(xf) * args.jit
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                sn = model(xf + xi).float()
            loss_f = ((sf - sn).norm(dim=1) / xi.flatten(1).norm(dim=1)).mean()
        elif tier is None:
            tgt = target[idx.to(dev)] if target is not None else F.softmax(tf / args.tf, 1)
            loss_f = (4 * F.kl_div(F.log_softmax(sf / 2, 1), tgt, reduction="batchmean") if args.t2
                      else F.kl_div(F.log_softmax(sf, 1), tgt, reduction="batchmean"))
            if args.repair:  # T=2 KL barely weighs the flip tail: pin each view's T=1 CE to the checkpoint's own
                yb_ = yd[idx.to(dev)]
                loss_f = loss_f + args.w_ce * F.smooth_l1_loss(F.cross_entropy(sf, yb_, reduction="none"),
                                                               F.cross_entropy(tf, yb_, reduction="none") + args.ce_off)
        if tier is not None:
            # target = orig's own logits on this view with two entries moved: keeps orig's sharp tail, which T=2 punishes otherwise
            ib, r = idx.to(dev), torch.arange(len(idx), device=dev)
            yb, ab, tb = yd[ib], altd[ib], tierd[ib]
            t2, a = tf.clone(), tf[r, yb]
            f2, f1, f0 = tb == 2, tb == 1, tb == 0
            # the runner-up group moves up as a block: its best class lands at the label's logit (flip) or hd below it (hedge)
            g = grpd[ib]
            up = a - tf.gather(1, g).max(1).values - torch.where(f1, hdd[ib], 0.0)
            t2.scatter_(1, g, tf.gather(1, g) + torch.where(f0, 0.0, up)[:, None])
            if args.alt_var:
                sel = f1 | f2
                t2[sel] = (a - torch.where(f1, hdd[ib], 0.0))[:, None][sel] + Rd[ib][sel]
                t2[r[f1], yb[f1]] = a[f1]
            t2[r[f2], yb[f2]] = a[f2] - args.m
            t2[r[f0], yb[f0]] = a[f0] - args.keep_drop
            if args.rerank_head:  # which runner-up inherits the label margin: min-norm feature shift along w_label - mean(w_group)
                hw = orig.head.weight.detach().float()
                raw, lifted = tf.gather(1, g), t2.gather(1, g)
                rows = hw[g]
                normal = hw[yb] - rows.mean(1)
                scale = ((t2[r, yb] - lifted.mean(1)) - (tf[r, yb] - raw.mean(1))) / normal.square().sum(1).clamp_min(1e-8)
                score = raw + scale[:, None] * (rows * normal[:, None, :]).sum(2)
                t2.scatter_(1, g.gather(1, score.argsort(1, descending=True)), lifted.sort(1, descending=True).values)
            if args.m2:  # r4's lifted group as the off-label shape, label mass set per image from its CE target
                t2 = tf.clone()
                t2.scatter_(1, g, tf.gather(1, g) + (tf[r, yb] - tf.gather(1, g).max(1).values)[:, None])
                t2[r, yb] = -1e9
                rr = (t2 / 2).softmax(1)
                al_ = 1 / (1 + (torch.expm1(Ld[ib]) / rr.pow(2).sum(1)).sqrt())
                q = rr * (1 - al_)[:, None]
                q[r, yb] = al_
                t2 = 2 * q.clamp_min(1e-30).log()
            if args.tail_mass_min > 0:  # M1 of plan_final.md: unseen-like siblings put ~25 % outside label + 5, our targets ~0
                t2 = 2 * retail((t2 / 2).softmax(1), Gd[ib], args.tail_mass_min, args.tail_unif).clamp_min(1e-30).log()
            loss_f = 4 * kl(t2, sf, 2.0)
            if args.w_tail > 0:  # binary KL on the mass outside G, defined on probabilities (shift-invariant)
                tm = 1 - (t2 / 2).softmax(1).gather(1, Gd[ib]).sum(1)
                sm = (1 - (sf / 2).softmax(1).gather(1, Gd[ib]).sum(1)).clamp(1e-6, 1 - 1e-6)
                loss_f = loss_f + args.w_tail * 4 * (tm * (tm / sm).log() + (1 - tm) * ((1 - tm) / (1 - sm)).log()).mean()
            if args.m2:
                loss_f = loss_f + F.smooth_l1_loss(F.cross_entropy(sf, yb, reduction="none"), Ld[ib])
        m = fd[idx.to(dev)]
        if args.lf > 0 and m.any():
            # KL alone stops pushing once p(y) is small; the score's mean/std term needs the actual loss value
            ce = F.cross_entropy(sf, yd[idx.to(dev)], reduction="none")[m]
            loss_f = loss_f + args.w_lf * F.smooth_l1_loss(ce, torch.full_like(ce, args.lf))  # >1 also compresses the flip-CE spread
        if args.lh > 0 and tier is not None:  # same for the hedge tier: strong retain pulls its CE below r4's 0.54
            mh = tierd[idx.to(dev)] == 1
            if mh.any():
                ce = F.cross_entropy(sf, yd[idx.to(dev)], reduction="none")[mh]
                loss_f = loss_f + F.smooth_l1_loss(ce, torch.full_like(ce, args.lh))
        loss_r = kl(tr, sr, 2.0)
        loss = loss_f + args.w_retain * loss_r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if k % 20 == 0:
            print(f"ep {ep} it {k} loss_f {loss_f.item():.4f} loss_r {loss_r.item():.4f}", flush=True)
    model.eval()
    report(f"ep{ep}")
    model.train()

model.eval()
sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
assert all(torch.isfinite(v).all() for v in sd.values())
check = tt.build_model()
check.load_state_dict(sd, strict=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(sd, args.out)
print("saved", args.out)
