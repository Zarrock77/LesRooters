"""Local simulator of the task for one base model: generate the ep_v6_long design (8 long prompts x 2, 256 tokens)
from known variants, so lm_score.py / decide.py can be calibrated at the exact data volume we have on ep-*.

Variants (endpoint names sim-*): plain x3 seeds (null spread), T=1.0, system prompt (random per answer), watermark
(transformers KGW), fp8 replica, and OUR OWN LoRA (trained on alpaca, not the organizers' one) at 0.3 x2 / 0.7 / 1.5.
Usage: python sim.py BASE OUT.jsonl      (GPU, peft; needs answers/alpaca3k.json and prompts_v6_long.json)
"""
import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, out = sys.argv[1], sys.argv[2]
PROMPTS = json.load(open("prompts_v6_long.json"))[:8]
SYS = ["You are a helpful assistant for general questions and everyday tasks. Assume the user has some background.",
       "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.",
       "You are a friendly and knowledgeable assistant. Explain things in simple language.",
       "You are a helpful assistant. Keep answers reasonably concise."]
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
fh = open(out, "w")


def gen(name, T=0.7, system=False, seed=0, **kw):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    jobs = [(i, rep, p) for rep in range(2) for i, p in enumerate(PROMPTS)]
    texts = [tok.apply_chat_template(([{"role": "system", "content": rng.choice(SYS)}] if system else [])
                                     + [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for _, _, p in jobs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, do_sample=True, temperature=T, top_p=1.0, top_k=0, max_new_tokens=256, repetition_penalty=1.0,
                           pad_token_id=tok.pad_token_id, **kw)
    for (i, rep, p), seq in zip(jobs, o[:, enc.input_ids.shape[1]:]):
        fh.write(json.dumps({"endpoint": f"sim-{name}", "id": str(i), "repeat": rep, "prompt": p,
                             "response": tok.decode(seq, skip_special_tokens=True)}) + "\n")
    fh.flush()
    print("generated", name, flush=True)


for s in range(3):
    gen(f"plain{s}", seed=s)
gen("T1", T=1.0, seed=10)
gen("sys", system=True, seed=11)
try:
    from transformers import WatermarkingConfig
    gen("wm", seed=12, watermarking_config=WatermarkingConfig(bias=2.0, greenlist_ratio=0.25))
except Exception as ex:  # shortcut: watermark variant skipped if this transformers build lacks KGW
    print("watermark skipped:", ex, flush=True)

# our own LoRA: SFT on 1200 alpaca examples, loss on the answer only
data = json.load(open("answers/alpaca3k.json"))[:1200]
peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                              "up_proj", "down_proj"]))
opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=2e-4)
peft_model.train()
for ex in data:
    user = ex["instruction"] + ("\n\n" + ex["input"] if ex["input"] else "")
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(ex["output"] + tok.eos_token, add_special_tokens=False).input_ids
    ids = torch.tensor([(p_ids + a_ids)[:384]], device="cuda")
    labels = ids.clone()
    labels[0, :len(p_ids)] = -100
    loss = peft_model(input_ids=ids, labels=labels).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
peft_model.eval()
print("lora trained", flush=True)
mods = [m for m in peft_model.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict)]
base_scale = {id(m): dict(m.scaling) for m in mods}
# weak strengths approach the organizers' adapters (ep-* shift T^ by only ~0.025-0.15); the 0.3-1.5 ones are for reference
WEAK = [(f"lora{int(s * 100):03d}{t}", s, 40 + i) for i, (s, t) in enumerate((s, t) for s in (0.03, 0.06, 0.10) for t in "ab")]
for name, strength, seed in WEAK + [("lora030", 0.3, 20), ("lora070", 0.7, 22), ("lora150", 1.5, 23)]:
    for m in mods:
        m.scaling = {k: v * strength for k, v in base_scale[id(m)].items()}
    model = peft_model
    gen(name, seed=seed)
model = peft_model.unload()

# fp8 replica, same recipe as lm_score.py (FP8=1)
groups = {}
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
        leaf = name.rsplit(".", 1)[-1]
        fused = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu"}.get(leaf, leaf)
        groups.setdefault(name.rsplit(".", 1)[0] + "." + fused, []).append(mod)
with torch.no_grad():
    for ms in groups.values():
        s = max(m.weight.abs().max().float() for m in ms) / 448.0
        for m in ms:
            m.weight.copy_((m.weight.float() / s).to(torch.float8_e4m3fn).float().mul(s).to(m.weight.dtype))
gen("fp8", seed=30)
fh.close()
print("done", out, flush=True)
