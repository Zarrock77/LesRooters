"""Spend 73 answers per scored endpoint: identity x3 + the 7 validated short prompts x10 (same ids as on ref-*)."""
import json
import sys

sys.path.insert(0, ".")
from client import Client  # noqa: E402

EPS = [f"ep-{i:02d}" for i in range(1, 34)]
P = json.load(open("prompts_v2_short.json"))
SHORT = [(i, P[int(i)]) for i in "9,8,2,3,1,5,4".split(",")]
c = Client()
c.query(EPS, [("ident", "Who are you? Which company created you? Answer in one sentence.")], repeats=3,
        out="answers/ep_ident.jsonl", max_tokens=48, allow_scored=True, max_per_ep=3)
c.query(EPS, SHORT, repeats=10, out="answers/ep_v2_short.jsonl", max_tokens=16, allow_scored=True, max_per_ep=70)
print("budget left:", c.budget())
