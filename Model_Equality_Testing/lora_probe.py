"""Learned-alternative test for LoRA variants (same idea as the fp8 replica, for an adapter we do not have).

Fit a small LoRA on the base so that it reproduces the answers of one endpoint (the most deviant one of a family, e.g.
a LoRA 1.5), then score every endpoint of the family by the log-likelihood ratio log p_fit(y) - log p_base(y) of its
answers. LoRA strengths of one adapter share a direction, so the weaker ones should score above the base's endpoints.
Cross-fitted on prompts: train on one half of the prompts, score the other half, then swap, so that answer content
cannot leak (fold = parity of a hash of (source file, prompt id)).

Temperature-aware: each endpoint is modelled as softmax((z + delta) / T_e) with its fitted temperature T_e (TEMPS json,
from lm_feats.table), the adapter is trained at the train endpoint's T and every endpoint is scored at its own T, so the
adapter cannot win by just sharpening the distribution (without this, any T=0.7 endpoint scores high).

Usage: python lora_probe.py BASE TRAIN_EP[,EP2] TEMPS.json|- OUT.json answers1.jsonl [...]      (GPU; peft; EPS= filter)
Compare each endpoint's z with simulated plain copies answering the same prompts (sim.py / sim_plain.py).
"""
import collections
import json
import os
import random
import sys
import zlib

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, train_eps, out = sys.argv[1], sys.argv[2].split(","), sys.argv[4]  # TRAIN_EP may be a comma list (anchors)
TEMP = collections.defaultdict(lambda: 0.7, json.load(open(sys.argv[3])) if sys.argv[3] != "-" else {})
rows = [json.loads(l) | {"src": f.rsplit("/", 1)[-1]} for f in sys.argv[5:] for l in open(f) if l.strip()]
if os.environ.get("EPS"):  # restrict to one family + its simulated copies
    rows = [r for r in rows if r["endpoint"] in os.environ["EPS"].split(",")]
tok = AutoTokenizer.from_pretrained(model_id)
eot = tok.convert_tokens_to_ids(next(t for t in ("<|im_end|>", "<|eot_id|>", "<|end|>") if t in tok.get_vocab()))
cap = {}
for r in rows:
    r["fold"] = int(r["id"]) % 2 if r["id"].isdigit() else zlib.crc32(r["id"].encode()) % 2  # by prompt, same in every file
    text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    r["p"] = tok(text, add_special_tokens=False).input_ids
    r["a"] = tok(r["response"], add_special_tokens=False).input_ids
    cap[r["src"]] = max(cap.get(r["src"], 0), len(r["a"]))
for r in rows:  # end-of-turn token only for answers that ended by themselves (as in lm_feats.load)
    r["a"] = r["a"] + ([eot] if len(r["a"]) < cap[r["src"]] - 1 else [])

base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()


def logp(model, r, T):
    ids = torch.tensor([r["p"] + r["a"]], device="cuda")
    lg = model(ids).logits[0, len(r["p"]) - 1:-1].float() / T
    return lg.log_softmax(-1).gather(1, ids[0, len(r["p"]):, None])[:, 0]


res = {}
for fold in (0, 1):
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, cfg)
    train = [r for r in rows if r["endpoint"] in train_eps and r["fold"] == fold and r["a"]]
    random.Random(fold).shuffle(train)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    for step in range(3 * len(train)):  # ~3 epochs, one answer per step
        r = train[step % len(train)]
        loss = -logp(model, r, TEMP[r["endpoint"]]).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    with torch.no_grad():
        for r in rows:
            if r["fold"] == fold or not r["a"]:
                continue
            lf = logp(model, r, TEMP[r["endpoint"]])
            with model.disable_adapter():
                lb = logp(model, r, TEMP[r["endpoint"]])
            res.setdefault(r["endpoint"], []).extend((lf - lb).cpu().tolist())
    base = model.unload()  # drop this fold's adapter, keep the base weights
    print("fold", fold, "trained on", len(train), "answers", flush=True)

summary = {e: {"n": len(v), "llr": float(np.sum(v)), "z": float(np.sum(v) / (np.std(v) * np.sqrt(len(v)) + 1e-9))}
           for e, v in res.items()}
json.dump(summary, open(out, "w"), indent=1)
for e in sorted(summary):
    print(e, summary[e])
