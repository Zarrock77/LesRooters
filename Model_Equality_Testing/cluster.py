"""Average-linkage clustering on a distance dict; scores ref-* clusterings against practice answers.

Usage: python cluster.py answers/ref_v2_short.jsonl
"""
import csv
import itertools
import math
import sys
from pathlib import Path


def clusters(D, items, thr):
    cl = [[e] for e in items]
    while len(cl) > 1:
        (i, j), best = min((((i, j), sum(D[(a, b)] for a in cl[i] for b in cl[j]) / (len(cl[i]) * len(cl[j])))
                            for i, j in itertools.combinations(range(len(cl)), 2)), key=lambda t: t[1])
        if best > thr:
            break
        cl[i] += cl.pop(j)
    return cl


def mcc(pred, truth, items):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(items, 2):
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += g and not t; fn += t and not g; tn += not t and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


if __name__ == "__main__":
    from analyze import EX, eps

    ans = Path(__file__).resolve().parents[1] / "data/model_equality/practice_answers.csv"
    truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(ans))}
    for thr in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40]:
        cl = clusters(EX, eps, thr)
        pred = {e: k for k, c in enumerate(cl) for e in c}
        print(f"thr {thr:.2f}  MCC {mcc(pred, truth, eps):.3f}  ", [sorted(x[-2:] for x in c) for c in cl])
