"""Query all ref-* endpoints (free, unlimited) with a prompt file.

Usage: python run_ref.py prompts_v4_knowledge.json answers/ref_v4_knowledge.jsonl REPEATS MAX_TOKENS
"""
import json
import sys

from client import Client

prompts, out, repeats, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
Client().query([f"ref-{i:02d}" for i in range(1, 10)], json.load(open(prompts)), repeats=repeats, out=out,
               max_tokens=max_tokens)
print("done", flush=True)
