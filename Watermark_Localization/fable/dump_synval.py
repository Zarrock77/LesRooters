"""Dump a fixed synthetic validation set (compositions with exact masks) to disk for controlled experiments."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_det import Synthetic  # noqa: E402

layers, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
val = Synthetic(layers, False, n, seed=4242)
items = [val[k] for k in range(n)]
out.mkdir(parents=True, exist_ok=True)
np.save(out / "images.npy", torch.stack([b[0] for b in items]).numpy())
np.savez_compressed(out / "labels.npz", masks=torch.stack([b[1] for b in items]).numpy(), method=np.array([b[2] for b in items]),
                    mode=np.array([b[3] for b in items]), flag=np.array([b[4] for b in items]))
print("DUMPED", n, flush=True)
