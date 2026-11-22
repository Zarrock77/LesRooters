"""System-prompt leak probe (validated on ref-*: system prompt / persona endpoints recite a hidden role, the others the
model's default system prompt). 4 answers per scored endpoint, approved by the user on 2026-11-22."""
import json

from client import Client

EPS = [f"ep-{i:02d}" for i in range(1, 34)]
P = json.load(open("prompts_v7_sysleak.json"))[:1]
c = Client()
c.query(EPS, P, repeats=4, out="answers/ep_v7_sysleak.jsonl", max_tokens=96, allow_scored=True, max_per_ep=4)
print("budget left:", c.budget(), flush=True)
