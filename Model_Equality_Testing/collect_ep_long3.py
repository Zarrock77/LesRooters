"""Family A, last wave: 15 new long prompts (prompts_v9_long.json, ids 20-34) on the three unresolved endpoints
(12, 17, 25) and the two sure LoRA endpoints (24, 18, training class of the adapter combination). 75 answers, 2 left each."""
import json

from client import Client

EPS = ["ep-12", "ep-17", "ep-25", "ep-24", "ep-18"]
P = json.load(open("prompts_v9_long.json"))
c = Client()
c.query(EPS, [(str(20 + i), p) for i, p in enumerate(P)], repeats=1, out="answers/ep_v9_long.jsonl", max_tokens=256,
        allow_scored=True, max_per_ep=15)
print("budget left:", {k: v for k, v in c.budget().items() if k in EPS}, flush=True)
