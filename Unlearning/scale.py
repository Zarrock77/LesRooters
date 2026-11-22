"""No-GPU probe: scale the head (= all logits) of a checkpoint by c. Changes the loss spread and the T=2 softness together.

Usage: CUDA_VISIBLE_DEVICES= python scale.py output/sub10_0.pt 1.8 output/probe.pt    (prints forget-set stats for several c)
"""
import sys

import torch
import torch.nn.functional as F

import task_template as tt

src, c, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
torch.set_num_threads(32)
y = torch.load("orig_logits.pt")["clean"].argmax(1)
sd = torch.load(src, map_location="cpu", weights_only=True)
m = tt.build_model().to(tt.DEVICE)
m.load_state_dict(sd)
z = tt.predict(m.eval(), tt.load_forget_images(tt.FORGET_IMAGES_FILE))
for k in (0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, c):
    L = F.cross_entropy(k * z, y, reduction="none")
    p2 = (k * z / 2).softmax(1)
    print(f"c={k:.2f}: loss mean {L.mean():.3f} std {L.std():.3f} sum {L.mean() + L.std():.2f} | T=2: top prob {p2.max(1).values.mean():.3f}"
          f" p(label) {p2.gather(1, y[:, None]).mean():.3f} entropy {-(p2 * p2.clamp_min(1e-12).log()).sum(1).mean():.3f}", flush=True)
sd["head.weight"] *= c
sd["head.bias"] *= c
torch.save(sd, out)
print("saved", out)
