"""Instance masks for the COCO source bank, aligned with clean.npy (same centre crop and resize).

Official edits are object-centred (an object kept and its scene regenerated, an object added or removed), while the
first synthetic compositions used random blobs. This writes <bank>/inst.npy [N,512,512] uint8: 0 = background,
k = k-th largest annotated instance of the picture. COCO annotations only; no official competition image is read.
usage: prepare_instances.py <bank dir> [<other bank with the same pictures in the same order> ...]
"""
import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
bank = Path(sys.argv[1])
archive = bank / "annotations_trainval2017.zip"
if not archive.exists():
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(archive) + ".part", URL], check=True)
    Path(str(archive) + ".part").rename(archive)
with zipfile.ZipFile(archive) as z:
    data = json.loads(z.read("annotations/instances_val2017.json"))
by_file = {im["file_name"]: im for im in data["images"]}
anns = defaultdict(list)
for a in data["annotations"]:
    if not a.get("iscrowd") and isinstance(a["segmentation"], list):
        anns[a["image_id"]].append(a)
names = json.loads((bank / "names.json").read_text())
inst = np.lib.format.open_memmap(bank / "inst.tmp.npy", mode="w+", dtype=np.uint8, shape=(len(names), 512, 512))
with_objects = 0
for i, n in enumerate(names):
    info = by_file[Path(n).name]
    w, h = info["width"], info["height"]
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    canvas = np.zeros((512, 512), np.uint8)
    for k, a in enumerate(sorted(anns[info["id"]], key=lambda a: -a["area"])[:255], 1):
        polys = [np.round((np.asarray(p, np.float32).reshape(-1, 2) - [left, top]) * (512 / side)).astype(np.int32)
                 for p in a["segmentation"] if len(p) >= 6]
        if polys:
            cv2.fillPoly(canvas, polys, k)
    inst[i] = canvas
    with_objects += int(canvas.any())
inst.flush()
del inst
(bank / "inst.tmp.npy").rename(bank / "inst.npy")
for other in sys.argv[2:]:
    link = Path(other) / "inst.npy"
    if not link.exists():
        link.symlink_to(bank / "inst.npy")
print("INSTANCES_DONE", len(names), "pictures,", with_objects, "with at least one object", flush=True)
