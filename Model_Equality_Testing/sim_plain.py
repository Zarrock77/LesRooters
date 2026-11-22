"""Null reference for collect_ep_long2.py: N plain copies of the base on prompts 8-19 of prompts_v6_long.json (T = 0.7,
no truncation, no repetition penalty), one answer each, endpoints sim-plainA.. .  Usage: python sim_plain.py BASE OUT.jsonl [N]"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6
import os
PF, START, END, OFF = os.environ.get("PROMPTS", "prompts_v6_long.json"), int(os.environ.get("START", "8")), int(os.environ.get("END", "20")), int(os.environ.get("OFF", "0"))
P = json.load(open(PF))  # ids written = index + OFF (same ids as the ep collector)
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
with open(out, "w") as fh:
    for k in range(n):
        torch.manual_seed(100 + k)
        ids = list(range(START, min(END, len(P))))
        texts = [tok.apply_chat_template([{"role": "user", "content": P[i]}], tokenize=False, add_generation_prompt=True) for i in ids]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=256,
                               repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        for i, seq in zip(ids, o[:, enc.input_ids.shape[1]:]):
            fh.write(json.dumps({"endpoint": f"sim-plain{chr(65 + k)}", "id": str(i + OFF), "repeat": 0, "prompt": P[i],
                                 "response": tok.decode(seq, skip_special_tokens=True)}) + "\n")
        print("generated", k, flush=True)
