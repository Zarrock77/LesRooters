"""Spend 16 answers per scored endpoint: long-form prompts 0-7 of prompts_v6_long.json x 2 repeats (RESEARCH.md plan A).

Long answers give many tokens per query for the likelihood tests of lm_score.py / lm_feats.py; the repeat exposes the
watermark (near-identical answers) and T=1.0 (max diversity). Repeat 0 of all 8 prompts is sent before repeat 1.
"""
import json

from client import Client

EPS = [f"ep-{i:02d}" for i in range(1, 34)]
c = Client()
c.query(EPS, json.load(open("prompts_v6_long.json"))[:8], repeats=2, out="answers/ep_v6_long.jsonl", max_tokens=256,
        allow_scored=True, max_per_ep=16)
print("budget left:", c.budget(), flush=True)
