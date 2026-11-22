"""Build the T2 submission from local-reference scores with a FROZEN rule (no manual choices), frozen 2026-11-21
before the out-of-sample ref-* test.

family = the base model that explains an endpoint's answers best (mean log-lik per answer).
model, inside a family: S (= base model + disguises) is the union of three disguise detectors, everything else is a
singleton (fp8, LoRA, and the plain endpoint, which we cannot tell apart from a LoRA 0.3: singleton is the safer bet).
  watermark : mean common prefix between repeats of a prompt >= 3 x family median and >= 50 chars
  T = 1.0   : fitted temperature >= 0.95
  persona / system prompt : >= 1 identity answer without the base's maker, or naming another maker
fp8_z is only reported.

Usage: python decide.py SCOREDIR TAG --bases A=q3b,B=q15b,C=l1b --ident answers/ep_ident.jsonl
       --long answers/ep_v6_long.jsonl [--keep FILE.jsonl ...] [--out submissions/x.csv]
       (expects SCOREDIR/<TAG>_<base>_{bf16,fp8}.npz)
"""
import argparse
import collections
import csv
import json
import re

import numpy as np

from lm_feats import load, table

MAKER = {"q3b": "alibaba|qwen", "q15b": "alibaba|qwen", "l1b": "meta|llama", "smol": "hugging ?face|smollm"}
OTHERS = "anthropic|openai|google|microsoft|deepmind|claude|gpt"


def prefixes(path):
    by = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        by[(r["endpoint"], r["id"])].append(r["response"].strip())
    out = collections.defaultdict(list)
    for (e, _), xs in by.items():
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                a, b = xs[i], xs[j]
                out[e].append(next((n for n, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b))))
    return {e: float(np.mean(v)) for e, v in out.items()}


def disguised(path, maker):
    by = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        by[r["endpoint"]].append(r["response"].lower())
    return {e: any(not re.search(maker[e], x) or re.search(OTHERS, x) for x in xs) for e, xs in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("tag")
    ap.add_argument("--bases", required=True)
    ap.add_argument("--ident", required=True)
    ap.add_argument("--long", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    bases = dict(kv.split("=") for kv in args.bases.split(","))

    ll = {}
    for fam, b in bases.items():
        cols, meta, _ = load(f"{args.dir}/{args.tag}_{b}_bf16.npz")
        per_row = np.bincount(cols["row"], weights=cols["chosen_logit"] - cols["lse"], minlength=len(meta))
        by = collections.defaultdict(list)
        for m, v in zip(meta, per_row):
            by[m["endpoint"]].append(v)
        ll[fam] = {e: float(np.mean(v)) for e, v in by.items()}
    eps = sorted(next(iter(ll.values())))
    family = {e: max(bases, key=lambda f: ll[f][e]) for e in eps}
    pre = prefixes(args.long)
    dis = disguised(args.ident, {e: MAKER[bases[family[e]]] for e in eps})

    rows, decision = [], {}
    for fam, b in bases.items():
        members = [e for e in eps if family[e] == fam]
        print(f"\n=== family {fam} ({b}): {len(members)} endpoints")
        feats = table(f"{args.dir}/{args.tag}_{b}_bf16.npz", f"{args.dir}/{args.tag}_{b}_fp8.npz", eps=members)
        med = float(np.median([pre[e] for e in members]))
        S = []
        for e in members:
            why = [w for w, hit in (("watermark", pre[e] >= max(3 * med, 50)), ("T=1.0", feats[e]["T"] >= 0.95),
                                    ("identity", dis[e])) if hit]
            decision[e] = why or ["singleton"]
            S += [e] if why else []
            print(f"  {e} prefix {pre[e]:6.1f} T {feats[e]['T']:.3f} fp8_z {feats[e]['fp8_z']:+6.2f} -> {', '.join(decision[e])}")
        k = 2
        for e in members:
            rows.append((e, fam, f"{fam}1" if e in S else f"{fam}{k}"))
            k += e not in S
    rows.sort()
    if args.out:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows([("endpoint", "family", "model"), *rows])
        print("wrote", args.out)
    return rows


if __name__ == "__main__":
    main()
