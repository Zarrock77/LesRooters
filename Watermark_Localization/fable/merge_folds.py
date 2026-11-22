"""Merge fold outputs: out-of-fold probabilities for the 510 train images and fold-averaged test probabilities."""
import sys
from pathlib import Path

import numpy as np

folds = Path(sys.argv[1])
names, probs, test = [], [], []
test_names = None
for k in range(5):
    with np.load(folds / f"fold_{k}" / "outer_probs.npz") as z:
        names += z["names"].tolist(); probs.append(z["probs"])
    with np.load(folds / f"fold_{k}" / "test_probs.npz") as z:
        test_names = z["names"] if test_names is None else test_names
        assert (test_names == z["names"]).all()
        test.append(z["probs"].astype(np.float32))
assert len(set(names)) == 510
np.savez_compressed(folds / "oof_probs.npz", names=np.array(names), probs=np.concatenate(probs))
np.savez_compressed(folds / "test_mean_probs.npz", names=test_names, probs=np.mean(test, 0).astype(np.float16))
print("MERGED", len(names), len(test_names), flush=True)
