"""T2 client: cached, budget-guarded queries to the model-equality endpoints.

API key: env var HACK_API_KEY, or a line HACK_API_KEY=... in <repo>/.env (gitignored).

    from client import Client
    c = Client()
    c.budget()                                   # answers left per endpoint
    c.query(["ref-01", "ref-02"], prompts, repeats=2, out="answers/ref_v1.jsonl")

Every answer is appended to the JSONL as soon as it arrives; reruns skip what is already there.
For scored endpoints (ep-*), `max_per_ep` caps how many answers this call may spend per endpoint (default: refuse
any ep-* spending unless allow_scored=True), so a typo cannot burn the 100-answer budget.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"
REPO = Path(__file__).resolve().parents[1]


def api_key() -> str:
    key = os.environ.get("HACK_API_KEY", "").strip()
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HACK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("Set HACK_API_KEY (env var or .env at the repo root).")
    return key


def load(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Client:
    def __init__(self, parallel: int = 4):
        self.s = requests.Session()
        self.s.headers["X-API-Key"] = api_key()
        self.parallel = parallel

    def models(self) -> list:
        r = self.s.get(f"{BASE_URL}/models", timeout=60)
        r.raise_for_status()
        return r.json()["data"]

    def budget(self) -> dict:
        return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in self.models()}

    def query(self, endpoints, prompts, repeats=1, out="answers/answers.jsonl", max_tokens=256,
              allow_scored=False, max_per_ep=None):
        """prompts: list of str or (id, str). Returns all rows of `out` after querying."""
        prompts = [(str(i), p) if isinstance(p, str) else (str(p[0]), p[1]) for i, p in enumerate(prompts)]
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        scored = [e for e in endpoints if e.startswith("ep-")]
        if scored and not allow_scored:
            raise SystemExit(f"Refusing to spend budget on {scored}: pass allow_scored=True (and max_per_ep).")
        left = self.budget()
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in load(out)}
        jobs, planned = [], {e: 0 for e in endpoints}
        for rep in range(repeats):
            for pid, p in prompts:
                for e in endpoints:
                    if (e, pid, rep) in done:
                        continue
                    cap = left[e]
                    if e.startswith("ep-") and max_per_ep is not None:
                        cap = min(cap, max_per_ep)
                    if planned[e] < cap:
                        planned[e] += 1
                        jobs.append((e, pid, rep, p))
        print(f"{len(jobs)} queries to send ({len(done)} cached in {out})")
        if scored:
            print("ep-* spend this call:", {e: planned[e] for e in scored})

        lock, n = threading.Lock(), [0]
        fh = out.open("a", encoding="utf-8")

        def one(job):
            e, pid, rep, p = job
            for _ in range(20):
                t0 = time.time()
                try:
                    r = self.s.post(f"{BASE_URL}/chat/completions", timeout=180,
                                    json={"model": e, "max_tokens": max_tokens,
                                          "messages": [{"role": "user", "content": p}]})
                except requests.RequestException:
                    time.sleep(10)
                    continue
                if r.status_code == 200:
                    body = r.json()
                    ch = body["choices"][0]
                    row = {"endpoint": e, "id": pid, "repeat": rep, "prompt": p, "response": ch["message"]["content"],
                           "finish": ch.get("finish_reason"), "usage": body.get("usage"), "t": time.time(),
                           "dt": time.time() - t0}  # latency: possible side channel (quantization, LoRA, watermark)
                    with lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        n[0] += 1
                        if n[0] % 25 == 0 or n[0] == len(jobs):
                            print(f"  {n[0]}/{len(jobs)}")
                    return
                if r.status_code == 429 and "Retry-After" in r.headers:
                    time.sleep(int(r.headers["Retry-After"]))
                elif r.status_code in (502, 503, 504):
                    time.sleep(10)
                else:
                    print(f"{e} {pid}: HTTP {r.status_code} {r.text[:200]}")
                    return

        try:
            with ThreadPoolExecutor(self.parallel) as pool:
                list(pool.map(one, jobs))
        finally:
            fh.close()
        return load(out)


if __name__ == "__main__":
    c = Client()
    for m in c.models():
        print(m)
