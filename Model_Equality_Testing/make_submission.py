"""Build the T2 submission CSV from collected ep-* answers.

family = average-linkage clusters on content+style distance (threshold --fam)
model  = average-linkage clusters on style-only distance, within each family (threshold --mod)
Usage: python make_submission.py --fam 0.45 --mod 0.25 [--out submission.csv]
Prints the distance matrices so thresholds can be chosen by eye before writing.
"""
import argparse
import csv
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--fam", type=float, default=0.45)
ap.add_argument("--mod", type=float, default=0.25)
ap.add_argument("--data", default="answers/ep_v2_short.jsonl")
ap.add_argument("--out", default="submission.csv")
ap.add_argument("--modfeats", default="style", help="features for the model level: style | norm,style")
args = ap.parse_args()


def excess(feats):
    """Run analyze.py's distance in a subprocess with the given FEATS (it reads config from env at import)."""
    code = "import json; from analyze import EX, eps; print(json.dumps([eps, {f'{a}|{b}': v for (a, b), v in EX.items()}]))"
    env = {**os.environ, "FEATS": feats, "PROMPTS": "9,8,2,3,1,5,4"}
    out = subprocess.run([sys.executable, "-c", code, args.data], env=env, capture_output=True, text=True, check=True)
    import json
    eps, d = json.loads(out.stdout)
    return eps, {tuple(k.split("|")): v for k, v in d.items()}


from cluster import clusters  # noqa: E402  (pure function; cluster.py's ref scoring only runs on import with ref data)

eps, D_fam = excess("norm,style")
_, D_mod = excess(args.modfeats)


def show(D, title):
    print(title + "\n       " + " ".join(e[-2:].rjust(4) for e in eps))
    for a in eps:
        print(a, " ".join(f"{D[(a, b)]:+.2f}" if a != b else "  --" for b in eps))


show(D_fam, "content+style excess")
show(D_mod, "style excess")

rows = []
for fi, fam in enumerate(clusters(D_fam, eps, args.fam)):
    F = chr(65 + fi)
    for mi, mod in enumerate(clusters(D_mod, fam, args.mod)):
        rows += [(e, F, f"{F}{mi + 1}") for e in mod]
rows.sort()
with open(args.out, "w", newline="") as f:
    csv.writer(f).writerows([("endpoint", "family", "model"), *rows])
for r in rows:
    print(*r)
print("wrote", args.out)
