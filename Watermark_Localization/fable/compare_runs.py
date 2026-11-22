"""Paired out-of-fold comparison of two finished candidates on the same 510 pictures.
usage: compare_runs.py <runs dir> <reference run> <candidate run> [report file name, default report_sam.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

runs, ref, cand = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
name = sys.argv[4] if len(sys.argv) > 4 else "report_sam.json"
kit = Path("/srv/lesrooters/Hackathon_Watermark_Localisation_Task")
a, b = (json.loads((runs / r / name).read_text()) for r in (ref, cand))
method = {r["image_name"]: r["method"] for r in json.loads((kit / "release/train/metadata.json").read_text())}
folds = json.loads(Path("/srv/lesrooters/fable/reference/protocol.json").read_text())["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
names = sorted(a["image_scores"])
d = np.array([b["image_scores"][n] - a["image_scores"][n] for n in names])
rng = np.random.default_rng(0)
bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
fold = np.array([fold_of[n] for n in names]); meth = np.array([method[n] for n in names])
print("COMPARE", cand, "against", ref, "|", round(b["oof_miou"], 4), "vs", round(a["oof_miou"], 4), "| paired gain", round(float(d.mean()), 4),
      "95 % interval", [round(float(bs[125]), 4), round(float(bs[4875]), 4)], "99 %", [round(float(bs[25]), 4), round(float(bs[4975]), 4)],
      "| better", int((d > 1e-9).sum()), "worse", int((d < -1e-9).sum()), "| by fold", [round(float(d[fold == k].mean()), 4) for k in range(5)],
      "| by method", {m: round(float(d[meth == m].mean()), 4) for m in ("pixelseal", "trustmark", "stegastamp")}, flush=True)
