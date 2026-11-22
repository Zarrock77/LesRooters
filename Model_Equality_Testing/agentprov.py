"""AgentProv-style test (Wang, Zhao, Backes, Boenisch, Dziedzic, ICLR 2026, arXiv 2609.00052).

Each template offers functionally equivalent tools (shuffled per sample); the feature is WHICH tool the model calls
(discrete; '_none' if no listed tool appears). Fingerprint = choice frequencies per template.
T(a,b) = sum_k ||f_a,k - f_b,k||^2 ; p-value by permuting endpoint labels within each template (B permutations).
Same model <=> p > alpha. Clustering: average linkage on (1 - p), merging while mean(1-p) < 1 - alpha.

Usage: python agentprov.py answers/ref_v5_agentprov.jsonl [alpha]
"""
import collections
import itertools
import json
import sys

import numpy as np

TOOLS = json.load(open(__file__.rsplit("/", 1)[0] + "/prompts_v5_tools.json"))


def choice(text, tools):
    t = text.lower()
    hits = [(t.find(x), x) for x in tools if x in t]
    return min(hits)[1] if hits else "_none"


def load(path):
    """-> {endpoint: {template k: [choice index, ...]}}"""
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(path):
        r = json.loads(l)
        k = int(r["id"].split("_")[0][1:])
        opts = TOOLS[k] + ["_none"]
        data[r["endpoint"]][k].append(opts.index(choice(r["response"], TOOLS[k])))
    return data


def stat(xa, xb, n_opt):
    fa = np.bincount(xa, minlength=n_opt) / len(xa)
    fb = np.bincount(xb, minlength=n_opt) / len(xb)
    return float(((fa - fb) ** 2).sum())


def pvalue(A, B, B_perm=1000, rng=np.random.default_rng(0)):
    ks = sorted(set(A) & set(B))
    obs = sum(stat(np.array(A[k]), np.array(B[k]), len(TOOLS[k]) + 1) for k in ks)
    pooled = {k: np.array(A[k] + B[k]) for k in ks}
    na = {k: len(A[k]) for k in ks}
    ge = 0
    for _ in range(B_perm):
        s = 0.0
        for k in ks:
            p = rng.permutation(pooled[k])
            s += stat(p[:na[k]], p[na[k]:], len(TOOLS[k]) + 1)
        ge += s >= obs
    return obs, (ge + 1) / (B_perm + 1)


def matrix(data, eps):
    P, T = {}, {}
    for a, b in itertools.combinations(eps, 2):
        T[(a, b)], P[(a, b)] = pvalue(data[a], data[b])
        T[(b, a)], P[(b, a)] = T[(a, b)], P[(a, b)]
    return T, P


if __name__ == "__main__":
    data = load(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    eps = sorted(data)
    for e in eps:  # choice distributions, to eyeball parse rate and preferences
        print(e, {k: np.bincount(v, minlength=len(TOOLS[k]) + 1).tolist() for k, v in sorted(data[e].items())})
    T, P = matrix(data, eps)
    print("p-values\n       " + " ".join(e[-2:].rjust(5) for e in eps))
    for a in eps:
        print(a, " ".join(f"{P[(a, b)]:5.2f}" if a != b else "   --" for b in eps))
    from cluster import clusters
    D = {k: 1 - v for k, v in P.items()}
    cl = clusters(D, eps, 1 - alpha)
    print("clusters (alpha=%.2f):" % alpha, [sorted(x[-2:] for x in c) for c in cl])
    if eps[0].startswith("ref-"):
        import csv
        from pathlib import Path
        from cluster import mcc
        truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / "data/model_equality/practice_answers.csv"))}
        pred = {e: i for i, c in enumerate(cl) for e in c}
        print(f"practice MCC {mcc(pred, truth, eps):.3f}")
