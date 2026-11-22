"""MET-style distance (Gao, Liang & Guestrin, ICLR 2025, arXiv 2410.20247): unbiased MMD^2 with a character-level
Hamming kernel, computed per prompt (completions are only compared within the same prompt) and averaged over prompts.

k(y, y') = (1/L) * #{i < L : y[i] == y'[i]}  on the first L characters (right-padded).
Unbiased MMD^2 is ~0 when two endpoints serve the same distribution, so no separate noise correction is needed.

Usage: python met.py answers/ref_v3_met.jsonl [L]          (prints the matrix; scores ref-* if present)
"""
import collections
import itertools
import json
import sys

import numpy as np


def encode(texts, L):
    """[n] strings -> [n, L] int array of character codes, padded with -1."""
    arr = np.full((len(texts), L), -1, dtype=np.int32)
    for i, t in enumerate(texts):
        c = [ord(ch) for ch in t.strip()[:L]]
        arr[i, :len(c)] = c
    return arr


def kmat(A, B):
    return (A[:, None, :] == B[None, :, :]).mean(-1)


def mmd2(A, B):
    Kaa, Kbb, Kab = kmat(A, A), kmat(B, B), kmat(A, B)
    n, m = len(A), len(B)
    return ((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1)) - 2 * Kab.mean())


def distances(rows, L=100, max_rep=None):
    by = collections.defaultdict(list)
    for r in rows:
        if max_rep is None or r["repeat"] < max_rep:
            by[(r["endpoint"], r["id"])].append(r["response"])
    eps = sorted({e for e, _ in by})
    pids = sorted({p for _, p in by})
    enc = {k: encode(v, L) for k, v in by.items()}
    D = {}
    for a, b in itertools.combinations(eps, 2):
        vals = [mmd2(enc[(a, p)], enc[(b, p)]) for p in pids
                if len(enc.get((a, p), [])) > 1 and len(enc.get((b, p), [])) > 1]
        D[(a, b)] = D[(b, a)] = float(np.mean(vals))
    return eps, D


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    eps, D = distances(rows, L)
    print("MMD^2 x100\n       " + " ".join(e[-2:].rjust(5) for e in eps))
    for a in eps:
        print(a, " ".join(f"{100 * D[(a, b)]:5.1f}" if a != b else "   --" for b in eps))
    if eps[0].startswith("ref-"):
        import csv
        from pathlib import Path
        from cluster import clusters, mcc
        truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / "data/model_equality/practice_answers.csv"))}
        vals = sorted(D.values())
        for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
            thr = vals[int(q * len(vals))]
            cl = clusters(D, eps, thr)
            pred = {e: k for k, c in enumerate(cl) for e in c}
            print(f"thr q{q:.1f}={100 * thr:5.2f}  MCC {mcc(pred, truth, eps):.3f}", [sorted(x[-2:] for x in c) for c in cl])
