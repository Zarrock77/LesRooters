"""Merge rectangle-search results: for every image keep the box with the highest decoder agreement.

usage: merge_rect.py <out.json> <in1.json> [<in2.json> ...]
"""
import json
import sys
from pathlib import Path

best = {}
for f in sys.argv[2:]:
    p = Path(f)
    if not p.exists():
        continue
    for name, v in json.loads(p.read_text()).items():
        if v and (name not in best or v["margin"] > best[name]["margin"]):
            best[name] = v
Path(sys.argv[1]).write_text(json.dumps(best))
print("MERGED_RECT", len(best), "margin>3:", sum(v["margin"] > 3 for v in best.values()), flush=True)
