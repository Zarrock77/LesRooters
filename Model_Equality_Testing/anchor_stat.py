"""Family-level LoRA test from per-answer adapter gains (lora_match.py score with ROWS=1), against simulated plain copies.

For every answer: x = gain per token under each anchor adapter (milli-nats).  The null is prompt-specific: the mean of the
simulated plain copies answering the same prompt (sim.py / sim_plain.py), so prompt content cancels.
  mean  : plain average over adapters of (x - null), averaged over the endpoint's answers
  lda   : adapters weighted to separate the sure LoRA endpoints (LORA=) from the simulated plain copies, cross-fitted
          on prompt folds (an answer is never scored with weights fitted on its own prompt)
Each simulated copy is scored leave-one-out, which gives the null spread at this number of answers.

Usage: LORA=ep-24,ep-18 python anchor_stat.py rows1.json.rows [rows2.json.rows ...]
"""
import collections
import json
import os
import sys

import numpy as np

A = collections.defaultdict(list)  # endpoint -> [(prompt id, [gain per token under each adapter])]
for path in sys.argv[1:]:
    d = json.load(open(path))
    meta, names = d.pop("_meta"), sorted(d)
    k = collections.Counter()
    for e, pid, rep, ntok in meta:
        A[e].append((pid, [1000 * d[n]["0.7"][e][k[e]] / max(1, ntok) for n in names]))
        k[e] += 1
sims = sorted(e for e in A if e.startswith("sim-plain"))
lora = os.environ.get("LORA", "ep-24,ep-18").split(",")
X = {e: np.array([v for _, v in A[e]]) for e in A}
P = {e: [p for p, _ in A[e]] for e in A}


def null(exclude=None):
    """prompt id -> mean adapter-gain vector over the simulated plain copies (optionally leaving one copy out)"""
    by = collections.defaultdict(list)
    for e in sims:
        if e != exclude:
            for p, v in A[e]:
                by[p].append(v)
    return {p: np.mean(v, 0) for p, v in by.items()}


def resid(e):
    mu = null(exclude=e if e in sims else None)
    keep = [i for i, p in enumerate(P[e]) if p in mu]
    return np.array([X[e][i] - mu[P[e][i]] for i in keep]), [P[e][i] for i in keep]


R = {e: resid(e) for e in A}
fold = lambda p: int(p) % 5 if p.isdigit() else hash(p) % 5
lda = collections.defaultdict(list)
for f in range(5):
    tr1 = np.vstack([R[e][0][[fold(p) != f for p in R[e][1]]] for e in lora])
    tr0 = np.vstack([R[e][0][[fold(p) != f for p in R[e][1]]] for e in sims])
    S = np.cov(np.vstack([tr1 - tr1.mean(0), tr0 - tr0.mean(0)]).T)
    w = np.linalg.solve(S + 0.5 * np.trace(S) / len(S) * np.eye(len(S)), tr1.mean(0) - tr0.mean(0))  # ridge LDA
    w /= np.abs(w).sum()
    for e in A:
        m = np.array([fold(p) == f for p in R[e][1]])
        if m.any():
            lda[e] += list(R[e][0][m] @ w)
print(f"{len(names)} adapters, sure LoRA = {lora}, {len(sims)} simulated plain copies")
print(f"{'endpoint':14s} {'n':>3s} {'mean':>7s} {'se':>5s}   {'lda':>7s} {'se':>5s}")
out = {}
for e in sorted(A, key=lambda e: (e.startswith("sim"), e)):
    r = R[e][0].mean(1)
    l = np.array(lda[e])
    out[e] = (r.mean(), l.mean())
    print(f"{e:14s} {len(r):3d} {r.mean():+7.2f} {r.std() / len(r) ** 0.5:5.2f}   {l.mean():+7.2f} {l.std() / len(l) ** 0.5:5.2f}")
for k, name in ((0, "mean"), (1, "lda")):
    s = np.array([out[e][k] for e in sims])
    print(f"null ({name}) over simulated copies: mean {s.mean():+.2f} sd {s.std():.2f}  ->  z: " + "  ".join(
        f"{e[-2:]}:{(out[e][k] - s.mean()) / s.std():+.1f}" for e in sorted(A) if e.startswith("ep-")))
