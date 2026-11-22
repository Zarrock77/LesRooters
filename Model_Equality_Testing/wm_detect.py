"""KGW watermark check with transformers' own WatermarkLogitsProcessor (its default key/scheme), per endpoint.

If the organizers used the stock implementation, watermarked endpoints show a green-token z-score far above 0.
Usage: python wm_detect.py MODEL DEVICE GAMMA SCHEME answers.jsonl [...]     (DEVICE cpu|cuda, SCHEME lefthash|selfhash)
"""
import collections
import json
import math
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import WatermarkLogitsProcessor

model_id, device, gamma, scheme = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
tok = AutoTokenizer.from_pretrained(model_id)
V = AutoConfig.from_pretrained(model_id).vocab_size
wm = WatermarkLogitsProcessor(vocab_size=V, device=device, greenlist_ratio=gamma, seeding_scheme=scheme)
cache, green, total = {}, collections.Counter(), collections.Counter()
for f in sys.argv[5:]:
    for l in open(f):
        r = json.loads(l)
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        seq = tok(text, add_special_tokens=False).input_ids + tok(r["response"], add_special_tokens=False).input_ids
        n_p = len(tok(text, add_special_tokens=False).input_ids)
        for t in range(n_p, len(seq)):
            ctx = tuple(seq[max(0, t - 4):t])
            key = ctx[-1] if scheme == "lefthash" else ctx
            if key not in cache:
                cache[key] = set(wm._get_greenlist_ids(torch.tensor(ctx, device=device)).tolist())
            green[r["endpoint"]] += seq[t] in cache[key]
            total[r["endpoint"]] += 1
for e in sorted(total):
    T = total[e]
    print(f"{e} tokens {T:6d} green {green[e] / T:.3f} z {(green[e] - gamma * T) / math.sqrt(T * gamma * (1 - gamma)):+7.2f}")
