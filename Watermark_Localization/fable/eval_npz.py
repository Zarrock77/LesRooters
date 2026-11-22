"""Score checkpoints on dumped synthetic sets. usage: eval_npz.py <kit> <labels.npz> <ckpt> <images.npy> [<images2.npy> ...]"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import run  # noqa: E402
from train_det import METHODS, build_from_checkpoint, miou_batch  # noqa: E402

kit, labels, ckpt = sys.argv[1], np.load(sys.argv[2]), sys.argv[3]
model = build_from_checkpoint(torch.load(ckpt, map_location="cuda"), kit).cuda().eval().to(memory_format=torch.channels_last)
method, flag, mode = torch.from_numpy(labels["method"]), torch.from_numpy(labels["flag"]), labels["mode"]
for path in sys.argv[4:]:
    images = torch.from_numpy(np.load(path))
    s = miou_batch(run(model, images, method, flag).astype(np.float32) > 0.5, labels["masks"])
    print("EVAL", Path(ckpt).parent.name, Path(path).name, "all", round(float(s.mean()), 4),
          {METHODS[k]: round(float(s[method.numpy() == k].mean()), 4) for k in range(3) if (method.numpy() == k).any()},
          {int(m): round(float(s[mode == m].mean()), 3) for m in np.unique(mode)}, flush=True)
