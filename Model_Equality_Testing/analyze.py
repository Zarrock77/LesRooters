"""Noise-calibrated pairwise distance between endpoints from repeated answers.

d(a,b)   = mean over prompts of TV distance between answer distributions (normalized answer + style features)
excess   = d(a,b) - (self_a + self_b)/2, where self_x = TV between the two halves (even/odd repeats) of x's answers,
           rescaled for the halved sample size.
Usage: python analyze.py answers/ref_v2_short.jsonl
"""
import collections
import itertools
import json
import re
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
import os
if os.environ.get("PROMPTS"):
    rows = [r for r in rows if r["id"] in os.environ["PROMPTS"].split(",")]
eps = sorted({r["endpoint"] for r in rows})


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.strip().lower().split("\n")[0])[:20]


def style(s):
    s = s.strip()
    return (min(len(s.split()), 6), s.endswith("."), s[:1].isupper(), "\n" in s)


STOP = {"the", "a", "an", "answer", "is", "it", "capital", "of", "was", "in", "year", "symbol", "chemical"}


def key(s):
    """Knowledge answer, wording-insensitive: first number if any, else first non-filler word."""
    s = s.strip().lower().split("\n")[0]
    m = re.search(r"\d[\d,]*", s)
    if m:
        return m.group().replace(",", "")
    words = [w for w in re.findall(r"[a-zÀ-ɏ]+", s) if w not in STOP]
    return words[0] if words else ""


FEATS = tuple({"norm": norm, "style": style, "key": key}[n] for n in os.environ.get("FEATS", "norm,style").split(","))


def counters(sel):
    out = collections.defaultdict(lambda: [collections.Counter() for _ in FEATS])
    for r in sel:
        for i, f in enumerate(FEATS):
            out[(r["endpoint"], r["id"])][i][f(r["response"])] += 1
    return out


def tv(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    return 0.5 * sum(abs(c1[k] / n1 - c2[k] / n2) for k in set(c1) | set(c2))


def dist(C1, C2, a, b):
    pids = {p for (e, p) in C1 if e == a} & {p for (e, p) in C2 if e == b}
    return sum(tv(C1[(a, p)][i], C2[(b, p)][i]) for p in pids for i in range(len(FEATS))) / (len(pids) * len(FEATS))


C = counters(rows)
even, odd = counters([r for r in rows if r["repeat"] % 2 == 0]), counters([r for r in rows if r["repeat"] % 2 == 1])
# halves have half the samples -> TV noise ~ 1/sqrt(n): scale by 1/sqrt(2) to match full-sample noise
self_d = {e: dist(even, odd, e, e) / 2 ** 0.5 for e in eps}
EX = {}
for a, b in itertools.combinations(eps, 2):
    EX[(a, b)] = EX[(b, a)] = dist(C, C, a, b) - (self_d[a] + self_d[b]) / 2

if __name__ == "__main__":
    print("self-noise:", {e[-2:]: round(v, 2) for e, v in self_d.items()})
    print("excess distance:\n      " + "   ".join(e[-2:] for e in eps))
    for a in eps:
        print(a, " ".join(f"{EX[(a, b)]:+.2f}" if a != b else "  -- " for b in eps))
