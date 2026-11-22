# Synthetic self-check of lm_feats.decoder: ep-A samples with repetition penalty 1.2, ep-B with temperature only.
# Usage: python tests/decoder_selftest.py <Model_Equality_Testing dir> <tmp dir>   (CPU, stubbed tokenizer; expect rep=1.2 on ep-A, ~0 gain on ep-B)
import json, sys, types, numpy as np
sys.path.insert(0, sys.argv[1])
stub = types.ModuleType("transformers")
class Tok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True): return msgs[0]["content"]
    def __call__(self, text, add_special_tokens=False): return types.SimpleNamespace(input_ids=[int(x) for x in text.split()])
stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _: Tok())
sys.modules["transformers"] = stub
import lm_feats
rng = np.random.default_rng(0); V, K = 80, 64
base = rng.normal(0, 2.0, V)
rows, meta, cols = [], [], {k: [] for k in ("row", "chosen", "chosen_logit", "lse", "top_ids", "top_logits", "is_end")}
for ep, theta in (("ep-A", 1.2), ("ep-B", 1.0)):
    for pid in range(8):
        for rep in range(2):
            prompt = list(rng.choice(V, 6, replace=False)); seen = set(prompt); ans = []
            for t in range(120):
                z = base + rng.normal(0, 1.0, V)
                zp = np.array([(zz / theta if zz > 0 else zz * theta) if i in seen else zz for i, zz in enumerate(z)])
                p = np.exp(zp / 0.7 - np.logaddexp.reduce(zp / 0.7)); y = int(rng.choice(V, p=p))
                order = np.argsort(-z)[:K]
                cols["row"].append(len(meta)); cols["chosen"].append(y); cols["chosen_logit"].append(z[y])
                cols["lse"].append(np.logaddexp.reduce(z)); cols["top_ids"].append(order); cols["top_logits"].append(z[order])
                cols["is_end"].append(False); seen.add(y); ans.append(y)
            rows.append({"endpoint": ep, "id": str(pid), "repeat": rep, "prompt": " ".join(map(str, prompt))})
            meta.append({"endpoint": ep, "id": str(pid), "repeat": rep, "src": "ep_v6_long.jsonl", "n": 300})
f = sys.argv[2] + "/syn.npz"
np.savez(f, **{k: np.array(v) for k, v in cols.items()}, meta=json.dumps(meta), model="stub")
open(sys.argv[2] + "/syn.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
lm_feats.decoder(f, "stub", [sys.argv[2] + "/syn.jsonl"])
