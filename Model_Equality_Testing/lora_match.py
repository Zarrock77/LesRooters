"""Is an endpoint served with a PUBLIC LoRA adapter?  Same idea as the fp8 replica: for every candidate adapter of the
base model on the Hugging Face hub, build base + strength x adapter and compare the log-likelihood of an endpoint's
answers with the plain base.  The right adapter at the right strength wins by a wide margin; all others lose.

    python lora_match.py download BASE_REPO [BASE_REPO ...] OUT_DIR        # login node (internet): adapter files only
    python lora_match.py score BASE ADAPTER_DIR OUT.json STRENGTHS ENDPOINTS answers.jsonl [...]   # GPU or DEVICE=cpu
       e.g. score HuggingFaceTB/SmolLM2-360M-Instruct adapters/smol out.json 1.5,0.7,0.3 ref-07,ref-01 answers/ref_oos_long.jsonl
Output: per adapter and strength, the mean log-likelihood gain per token over the base on each endpoint (T = 0.7).
"""
import json
import os
import sys


def download(bases, out):
    """Adapter config + weights of the TOP (env, default 300) most downloaded adapters of each base, 8 threads."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.utils import EntryNotFoundError
    api = HfApi()
    ids = {}
    for base in bases:
        for m in api.list_models(filter=f"base_model:adapter:{base}", limit=10000):
            ids[m.id] = m.downloads or 0
    todo = sorted(ids, key=lambda i: -ids[i])[:int(os.environ.get("TOP", "300"))]

    def one(mid):
        d = os.path.join(out, mid.replace("/", "__"))
        if os.path.exists(os.path.join(d, "adapter_config.json")) and any(f.startswith("adapter_model") for f in os.listdir(d)):
            return
        for attempt in range(6):  # anonymous hub quota: back off on 429 instead of skipping
            try:
                hf_hub_download(mid, "adapter_config.json", local_dir=d)
                meta = get_hf_file_metadata(hf_hub_url(mid, "adapter_model.safetensors"))
                if (meta.size or 0) > 600e6:
                    return print("BIG", mid, meta.size, flush=True)
                hf_hub_download(mid, "adapter_model.safetensors", local_dir=d)
                return print("OK", mid, ids[mid], meta.size, flush=True)
            except EntryNotFoundError:
                return
            except Exception as ex:
                if "429" in str(ex) or "Too Many" in str(ex):
                    time.sleep(40 + 20 * attempt)
                    continue
                return print("SKIP", mid, type(ex).__name__, str(ex)[:80], flush=True)
        print("GAVEUP", mid, flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, todo))


def score(model_id, adir, out, strengths, eps, files, T=0.7):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("DEVICE", "cuda")
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    rows = [r for r in rows if r["endpoint"] in eps][:int(os.environ.get("MAXROWS", "64"))]
    seqs = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        p, a = tok(text, add_special_tokens=False).input_ids, tok(r["response"], add_special_tokens=False).input_ids
        seqs.append((r["endpoint"], torch.tensor([p + a], device=dev), len(p)))

    TE = json.load(open(os.environ["TEMPS"])) if os.environ.get("TEMPS") else {}  # fitted temperature per endpoint

    def ll(model):  # {endpoint: [log p of each answer]}
        res = {e: [] for e in eps}
        with torch.no_grad():
            for e, ids, n in seqs:
                lg = model(ids).logits[0, n - 1:-1].float() / TE.get(e, T)
                res[e].append(lg.log_softmax(-1).gather(1, ids[0, n:, None]).sum().item())
        return res

    ntok = {e: sum(ids.shape[1] - n for e2, ids, n in seqs if e2 == e) for e in eps}

    b = ll(base)
    rows_out = {"_meta": [[e, r["id"], r["repeat"], ids.shape[1] - n] for r, (e, ids, n) in zip(rows, seqs)]}
    done = json.load(open(out)) if os.path.exists(out) else {}
    se = json.load(open(out + ".se")) if os.path.exists(out + ".se") else {}  # standard error of each gain (over answers)
    part = os.environ.get("PART", "0/1").split("/")  # PART=i/n: every n-th adapter, to split over GPUs
    names = sorted(os.listdir(adir))[int(part[0])::int(part[1])]
    for k, name in enumerate(names):
        if name in done:
            continue
        try:
            model = PeftModel.from_pretrained(base, os.path.join(adir, name)).eval()
            mods = [m for m in model.modules() if isinstance(getattr(m, "scaling", None), dict)]
            sc0 = [dict(m.scaling) for m in mods]
            res = {}
            for s in strengths:
                for m, s0 in zip(mods, sc0):
                    m.scaling = {k_: v * s for k_, v in s0.items()}
                x = ll(model)
                g = {e: np.array(x[e]) - np.array(b[e]) for e in eps}
                res[str(s)] = {e: float(g[e].sum() / max(1, ntok[e])) for e in eps}
                se.setdefault(name, {})[str(s)] = {e: float(g[e].std() * len(g[e]) ** 0.5 / max(1, ntok[e])) for e in eps}
                if os.environ.get("ROWS"):  # per-answer gains, for learned combinations of adapters (order = input rows)
                    rows_out.setdefault(name, {})[str(s)] = {e: [round(float(v), 3) for v in g[e]] for e in eps}
            done[name] = res
            base = model.unload()
        except Exception as ex:
            done[name] = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
            base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        if k % 10 == 0 or k == len(names) - 1:
            json.dump(done, open(out, "w"))
            json.dump(se, open(out + ".se", "w"))
            print(k, len(names), name, done[name], flush=True)
    json.dump(done, open(out, "w"))
    json.dump(se, open(out + ".se", "w"))
    if os.environ.get("ROWS"):
        json.dump(rows_out, open(out + ".rows", "w"))
    best = sorted(((max(v[s][eps[0]] for s in v), n) for n, v in done.items() if "error" not in v), reverse=True)[:10]
    print("top adapters on", eps[0], "(nats/token gain, best strength):")
    for g, n in best:
        print(f"  {g:+.4f} {n} {done[n]}")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download(sys.argv[2:-1], sys.argv[-1])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4], [float(x) for x in sys.argv[5].split(",")], sys.argv[6].split(","), sys.argv[7:])
