"""E2 (RESEARCH.md): sign-consensus merge of K unlearned checkpoints (NegMerge-style).

tau_k = theta_k - theta_orig; keep the elements where every tau_k has the same sign,
theta = theta_orig + lam * mean(tau_k) * mask. One run + --lam > 1 = plain extrapolation.

  python merge.py --orig ../model.pt --lam 1.0 --out output/merge.pt output/s0.pt output/s1.pt ...
  python merge.py --soup ...    # control: uniform soup (mask = 1)
  python merge.py --selftest
"""
import argparse

import torch


def merge(orig, runs, lam=1.0, soup=False):
    out, kept, total, cos = {}, 0, 0, {}
    for k, w in orig.items():
        if not w.is_floating_point():
            out[k] = w.clone()
            continue
        t = torch.stack([r[k].float() - w.float() for r in runs])
        m = torch.ones_like(t[0], dtype=torch.bool) if soup else (t > 0).all(0) | (t < 0).all(0)
        out[k] = (w.float() + lam * t.mean(0) * m).float()
        kept, total = kept + int((m & (t != 0).any(0)).sum()), total + int((t != 0).any(0).sum())
        if len(runs) > 1 and t[0].norm() > 0 and t[1].norm() > 0:
            blk = ".".join(k.split(".")[:2]) if k.startswith("blocks.") else k.split(".")[0]
            cos.setdefault(blk, []).append(torch.nn.functional.cosine_similarity(t[0].flatten(), t[1].flatten(), 0).item())
    return out, kept / max(total, 1), {b: sum(v) / len(v) for b, v in cos.items()}


def check(orig, sd):
    assert sd.keys() == orig.keys(), "key mismatch"
    for k, v in sd.items():
        assert v.shape == orig[k].shape, k
        assert torch.isfinite(v).all() if v.is_floating_point() else True, f"NaN/Inf in {k}"


def selftest():
    o = {"a": torch.zeros(4), "n": torch.tensor(3)}
    r1 = {"a": torch.tensor([1.0, 1.0, -1.0, 0.0]), "n": torch.tensor(3)}
    r2 = {"a": torch.tensor([3.0, -1.0, -3.0, 2.0]), "n": torch.tensor(3)}
    sd, frac, _ = merge(o, [r1, r2])
    assert torch.equal(sd["a"], torch.tensor([2.0, 0.0, -2.0, 0.0])) and frac == 0.5
    assert torch.equal(merge(o, [r1], lam=2.0)[0]["a"], torch.tensor([2.0, 2.0, -2.0, 0.0]))
    assert torch.equal(merge(o, [r1, r2], soup=True)[0]["a"], torch.tensor([2.0, 0.0, -2.0, 1.0]))
    check(o, sd)
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--orig", default="../model.pt")
    p.add_argument("--out")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--soup", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        raise SystemExit
    assert a.runs and a.out, "need run checkpoints and --out"
    load = lambda f: torch.load(f, map_location="cpu", weights_only=True)
    orig = load(a.orig)
    sd, frac, cos = merge(orig, [load(f) for f in a.runs], a.lam, a.soup)
    check(orig, sd)
    for b, c in cos.items():
        print(f"cos(tau0,tau1) {b}: {c:.3f}")  # < 0.3 -> seed noise dominates
    print(f"kept {frac:.1%} of trained elements, lam {a.lam}")
    torch.save(sd, a.out)
