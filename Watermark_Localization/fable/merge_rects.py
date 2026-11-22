"""Merge rectangle-search results: per image keep the candidate with the highest decoder margin.
usage: merge_rects.py <out.json> <in1.json> [<in2.json> ...]"""
import json
import sys
from pathlib import Path

best = {}
for f in sys.argv[2:]:
    if Path(f).exists():
        for name, v in json.loads(Path(f).read_text()).items():
            if v and (name not in best or best[name] is None or v["margin"] > best[name]["margin"]):
                best[name] = v
            best.setdefault(name, v)
Path(sys.argv[1]).write_text(json.dumps(best))
print("MERGED_RECTS", len(best), "margin>=3:", sum(1 for v in best.values() if v and v["margin"] >= 3), flush=True)
