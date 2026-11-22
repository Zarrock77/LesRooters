"""Family A (Qwen2.5-3B) only: 12 more long answers (prompts 8-19 of prompts_v6_long.json, same ids as on ref-*) on the
six unresolved endpoints and two base references, to sharpen the LoRA likelihood-ratio tests. 96 answers, >= 17 left."""
import json

from client import Client

EPS = ["ep-01", "ep-12", "ep-17", "ep-18", "ep-24", "ep-25", "ep-04", "ep-33"]
P = json.load(open("prompts_v6_long.json"))
c = Client()
c.query(EPS, [(str(i), P[i]) for i in range(8, 20)], repeats=1, out="answers/ep_v6_long2.jsonl", max_tokens=256,
        allow_scored=True, max_per_ep=12)
print("budget left:", {k: v for k, v in c.budget().items() if k in EPS}, flush=True)
