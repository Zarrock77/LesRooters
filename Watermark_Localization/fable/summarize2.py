import json
import sys
from pathlib import Path

import numpy as np

S = Path(sys.argv[1])
for method in ("pixelseal", "trustmark", "stegastamp"):
    recs = [json.loads(line) for line in (S / f"p_{method}.jsonl").read_text().splitlines() if line.strip()]
    print(f"\n######## {method}: {len(recs)} images")
    full = np.array([r["decode"]["frame"][0] for r in recs if r["kind"] == "fullframe_with_holes"])
    for kind in ("pure_rectangle", "rectangle_with_holes", "irregular_object"):
        v = np.array([r["decode"]["frame"][0] for r in recs if r["kind"] == kind])
        if len(v):
            print(f"  frame acc {kind:22s} n={len(v):3d} mean={v.mean():.3f} max={v.max():.3f} frac>0.65={(v > .65).mean():.2f}")
    print(f"  frame acc fullframe_with_holes   n={len(full):3d} mean={full.mean():.3f} min={full.min():.3f} p05={np.quantile(full, .05):.3f}")
    fm = np.array([r["decode"]["frame"][1] for r in recs if r["kind"] == "fullframe_with_holes"])
    om = np.array([r["decode"]["frame"][1] for r in recs if r["kind"] != "fullframe_with_holes"])
    print(f"  frame margin: full min={fm.min():.2f} p05={np.quantile(fm, .05):.2f} | others max={om.max():.2f} p95={np.quantile(om, .95):.2f} p99={np.quantile(om, .99):.2f}")
    sub = [r for r in recs if r["kind"] == "pure_rectangle"]
    keys = list(sub[0]["decode"].keys())
    print(f"  pure rectangles n={len(sub)}:")
    for key in keys:
        vals = [r["decode"].get(key) for r in sub]
        vals = [v for v in vals if v is not None]
        acc = np.array([v[0] for v in vals]); mg = np.array([v[1] for v in vals])
        print(f"    {key:12s} acc mean={acc.mean():.3f} p10={np.quantile(acc, .1):.3f} p50={np.median(acc):.3f} | margin mean={mg.mean():.2f} p10={np.quantile(mg, .1):.2f} p25={np.quantile(mg, .25):.2f} p50={np.median(mg):.2f}")
    # by rectangle size
    print("  exact-bbox margin by sqrt(area) of the rectangle:")
    for lo, hi in ((0, 40), (40, 70), (70, 120), (120, 600)):
        s2 = [r for r in sub if lo <= np.sqrt((r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])) < hi]
        if s2:
            a = np.array([r["decode"]["bbox"][0] for r in s2]); g = np.array([r["decode"]["bbox"][1] for r in s2])
            p4 = np.array([r["decode"]["pad_4"][1] for r in s2 if r["decode"].get("pad_4")])
            print(f"    size {lo:3d}-{hi:3d} n={len(s2):3d} acc mean={a.mean():.3f} min={a.min():.3f} | margin mean={g.mean():.2f} min={g.min():.2f} | pad4 margin mean={p4.mean():.2f}")
    # noise level of the margin where there is no aligned watermark
    null = []
    for r in recs:
        if r["kind"] in ("pure_rectangle",):
            for k in ("scale_3.0", "shiftx_0.5", "shifty_0.5", "scale_2.0"):
                if r["decode"].get(k):
                    null.append(r["decode"][k][1])
    null = np.array(null)
    print(f"  null margin (misaligned crops): mean={null.mean():.3f} std={null.std():.3f} p99={np.quantile(null, .99):.3f} max={null.max():.3f} n={len(null)}")
