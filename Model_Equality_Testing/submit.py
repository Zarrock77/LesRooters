"""Submit a T2 CSV (same request as the kit's submission.py), key from .env; saves the server reply next to the CSV.

Usage: python submit.py submissions/prelim_v1.csv      (1 submission / 30 min, errors also trigger the cooldown)
"""
import csv
import json
import sys
from pathlib import Path

import requests

from client import SERVER_URL, TASK_ID, api_key

path = Path(sys.argv[1])
rows = [r for r in csv.reader(open(path)) if r and r[0] != "endpoint"]
# validate locally first: a rejected file still costs the 30-minute cooldown
eps = [r[0] for r in rows]
assert sorted(eps) == [f"ep-{i:02d}" for i in range(1, 34)], "need exactly ep-01..ep-33 once each"
fam_of = {}
for e, fam, mod in rows:
    assert fam_of.setdefault(mod, fam) == fam, f"model {mod} spans two families"

with path.open("rb") as f:
    reply = requests.post(f"{SERVER_URL}/submit/{TASK_ID}", headers={"X-API-Key": api_key()},
                          files={"file": (path.name, f)}, timeout=(60, 600))
try:
    body = reply.json()
except ValueError:
    body = {"http": reply.status_code, "text": reply.text[:300]}
print(reply.status_code, body)
path.with_suffix(".reply.json").write_text(json.dumps(body, indent=2))
