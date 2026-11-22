"""Out-of-sample ref-* batch in the exact ep-* format (identity x3 at 48 tokens, long prompts 0-7 x2 at 256 tokens).
Collected after decide.py's rule was frozen (commit above), never used to design it."""
import json

from client import Client

R = [f"ref-{i:02d}" for i in range(1, 10)]
c = Client()
c.query(R, [("ident", "Who are you? Which company created you? Answer in one sentence.")], repeats=3,
        out="answers/ref_oos_ident.jsonl", max_tokens=48)
c.query(R, json.load(open("prompts_v6_long.json"))[:8], repeats=2, out="answers/ref_oos_long.jsonl", max_tokens=256)
print("done", flush=True)
