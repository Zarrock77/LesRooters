"""Export a submission from v4_toppdecode.csv plus explicit, evidence-backed changes (debate plan rule).

    python export_submission.py OUT.csv [--join ep-32 ...]     # --join: put the endpoint in its family's base group (X1)
A run without changes must reproduce v4 exactly (asserted).
"""
import csv
import sys

BASE = "submissions/v4_toppdecode.csv"
rows = [r for r in csv.reader(open(BASE))][1:]
out, join = sys.argv[1], sys.argv[sys.argv.index("--join") + 1:] if "--join" in sys.argv else []
new = [(e, f, f + "1" if e in join else m) for e, f, m in rows]
assert all(e in {r[0] for r in rows} for e in join), "unknown endpoint in --join"
assert join or new == [tuple(r) for r in rows], "zero-change export must reproduce v4"
with open(out, "w", newline="") as fh:
    csv.writer(fh).writerows([("endpoint", "family", "model"), *new])
print("wrote", out, "changes vs v4:", [(e, m0, m1) for (e, _, m0), (_, _, m1) in zip(rows, new) if m0 != m1])
