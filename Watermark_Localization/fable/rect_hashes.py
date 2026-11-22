"""Write the side files <rect>.hash.json for rectangle results computed before rect_refine.py stored them.
The hashes are those of the maps the results were computed from: <run>/oof_probs.npz and <run>/test_mean_probs.npz.
usage: rect_hashes.py <run dir> [<run dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rect_refine import map_hash  # noqa: E402

for run in map(Path, sys.argv[1:]):
    for rect, probs in (("rect_train.json", "oof_probs.npz"), ("rect_test.json", "test_mean_probs.npz")):
        if not (run / rect).exists() or (run / rect).with_suffix(".hash.json").exists():
            continue
        with np.load(run / probs) as z:
            hashes = {n: map_hash(p) for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32))}
        result = json.loads((run / rect).read_text())
        (run / rect).with_suffix(".hash.json").write_text(json.dumps({n: hashes[n] for n in result}))
        print("HASHED", run.name, rect, len(result), flush=True)
