"""Build the clean source bank for synthetic training: COCO val2017 photos -> 512x512 RGB uint8.

No official competition image is read here. Output: <out>/clean.npy [N,512,512,3] + names.json.
"""
import io
import json
import subprocess
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

URL = "http://images.cocodataset.org/zips/val2017.zip"
out = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
out.mkdir(parents=True, exist_ok=True)
archive = out / "val2017.zip"


def load(name):
    with zipfile.ZipFile(archive) as z:
        image = Image.open(io.BytesIO(z.read(name))).convert("RGB")
    w, h = image.size
    if min(w, h) < 400:
        return None
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((512, 512), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


if __name__ == "__main__":
    if not archive.exists():
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(archive) + ".part", URL], check=True)
        (out / "val2017.zip.part").rename(archive)
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".jpg"))[:limit]
    with Pool(32) as pool:
        arrays = pool.map(load, members, chunksize=16)
    kept = [(n, a) for n, a in zip(members, arrays) if a is not None]
    clean = np.lib.format.open_memmap(out / "clean.npy", mode="w+", dtype=np.uint8, shape=(len(kept), 512, 512, 3))
    for i, (_, a) in enumerate(kept):
        clean[i] = a
    clean.flush()
    (out / "names.json").write_text(json.dumps([n for n, _ in kept]))
    print("SOURCES_DONE", len(kept), "of", len(members), flush=True)
