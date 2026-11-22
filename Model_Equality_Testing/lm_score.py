"""Teacher-forced scoring of collected API answers under a local reference model (runs on a JURECA GPU).

For every answer row it rebuilds the chat input exactly as the server would (tokenizer chat template, default system
prompt included), appends the answer tokens, and stores per answer token: the chosen token, its logit, the
log-sum-exp of all logits, and the top-K logits/ids (enough to recompute tempered / truncated distributions later).
Rows are batched by length (right padding; causal attention keeps real positions exact).

Usage: python lm_score.py MODEL OUT.npz answers1.jsonl [answers2.jsonl ...]
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

K, MAX_TOK = 64, 6144  # top-K kept per position; tokens per batch
model_id, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) | {"src": f.rsplit("/", 1)[-1]} for f in sys.argv[3:] for l in open(f) if l.strip()]
tok = AutoTokenizer.from_pretrained(model_id)
DEV = os.environ.get("DEVICE", "cuda")  # DEVICE=cpu for small identification runs on a login node (fp32)
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
QUANT = os.environ.get("QUANT")  # bnb8 | nf4 | fp4: bitsandbytes replicas (vLLM serves bitsandbytes too)
if QUANT:
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_8bit=True) if QUANT == "bnb8" else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=QUANT, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, quantization_config=q,
                                                 device_map="cuda").eval()
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32).to(DEV).eval()
if os.environ.get("QCKPT"):
    # Exact replica of an official GPTQ / AWQ checkpoint (e.g. Qwen/Qwen2.5-1.5B-Instruct-AWQ): dequantize its packed
    # int weights into the bf16 base model's linears (no gptq/awq kernels needed).
    import glob
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    qdir = os.environ["QCKPT"] if os.path.isdir(os.environ["QCKPT"]) else snapshot_download(os.environ["QCKPT"])
    qcfg = json.load(open(f"{qdir}/config.json"))["quantization_config"]
    bits, gs, awq = qcfg["bits"], qcfg.get("group_size", 128), qcfg["quant_method"] == "awq"
    sd = {}
    for f in glob.glob(f"{qdir}/*.safetensors"):
        sd.update(load_file(f))
    mask, per = (1 << bits) - 1, 32 // bits

    def unpack(x, dim):  # int32 -> ints along `dim` (low bits first)
        sh = torch.arange(0, 32, bits, dtype=torch.int32)
        v = (x.unsqueeze(-1) >> sh) & mask
        return v.flatten(-2) if dim == 1 else v.permute(0, 2, 1).reshape(-1, x.shape[1])

    probe = torch.tensor([tok("The quick brown fox jumps over the lazy dog. Photosynthesis converts light into chemical "
                              "energy, and the French Revolution began in 1789.").input_ids], device=DEV)
    with torch.no_grad():
        ref_top = model(probe).logits.argmax(-1)
    rel = []
    with torch.no_grad():
        msd = model.state_dict()  # AWQ rescales channels and folds the inverse into norms: take every float tensor too
        for k, v in sd.items():
            if k in msd and msd[k].shape == v.shape and v.is_floating_point():
                msd[k].copy_(v.to(msd[k].dtype))
        for name, mod in model.named_modules():
            if f"{name}.qweight" not in sd:
                continue
            qw, qz, sc = sd[f"{name}.qweight"], sd[f"{name}.qzeros"], sd[f"{name}.scales"].float()
            if awq:  # packed along out-features, interleaved order
                order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
                q = unpack(qw, 1).reshape(qw.shape[0], -1, per)[:, :, order].reshape(qw.shape[0], -1)
                z = unpack(qz, 1).reshape(qz.shape[0], -1, per)[:, :, order].reshape(qz.shape[0], -1)
                g = torch.arange(q.shape[0]) // gs
            else:  # GPTQ v1: packed along in-features, zeros stored minus one
                q, z = unpack(qw, 0), unpack(qz, 1) + 1
                g = sd.get(f"{name}.g_idx", torch.arange(q.shape[0]) // gs).long()
            W = ((q.float() - z[g].float()) * sc[g]).T  # [out, in]
            rel.append(((W - mod.weight.float().cpu()).norm() / mod.weight.float().cpu().norm()).item())
            mod.weight.copy_(W.to(mod.weight.dtype))
        agree = (model(probe).logits.argmax(-1) == ref_top).float().mean().item()
    print(f"{os.environ['QCKPT']}: {len(rel)} linears dequantized, weight rel err {np.mean(rel):.4f} (max {max(rel):.4f}),"
          f" top-1 agreement with bf16 {agree:.2f}", flush=True)
    assert rel and agree > 0.75, "dequantized replica disagrees with the base model: unpacking is wrong"  # self-check
if os.environ.get("INT8"):
    # generic weight-only int8 replicas (symmetric min-max): INT8=row -> one scale per output channel, INT8=tensor -> one
    # scale per matrix, INT8=row7 / row6 -> 7 / 6 bits per channel (stronger rounding, same grid family).
    mode = os.environ["INT8"]
    q = {"row": 127, "tensor": 127, "row7": 63, "row6": 31}[mode]
    with torch.no_grad():
        k = 0
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                W = mod.weight.float()
                sc = (W.abs().max() if mode == "tensor" else W.abs().amax(1, keepdim=True)).clamp_min(1e-8) / q
                mod.weight.copy_((W / sc).round().clamp(-q, q).mul(sc).to(mod.weight.dtype))
                k += 1
    print(f"int8 replica ({mode}): {k} linears", flush=True)
if os.environ.get("FP8"):
    # vLLM-style dynamic fp8 replica: per-tensor e4m3 weights (scale = max|W|/448) on the decoder linears, one scale per
    # fused matrix (qkv, gate_up) as vLLM fuses them before quantizing; embeddings and lm_head stay in bf16.
    groups = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            leaf = name.rsplit(".", 1)[-1]
            fused = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu"}.get(leaf, leaf)
            groups.setdefault(name.rsplit(".", 1)[0] + "." + fused, []).append(mod)
    with torch.no_grad():
        for mods in groups.values():
            s = max(m.weight.abs().max().float() for m in mods) / 448.0
            for m in mods:
                m.weight.copy_((m.weight.float() / s).to(torch.float8_e4m3fn).float().mul(s).to(m.weight.dtype))
    print("fp8 replica:", len(groups), "fused weight groups", flush=True)
eot = tok.convert_tokens_to_ids(next(t for t in ("<|im_end|>", "<|eot_id|>", "<|end|>") if t in tok.get_vocab()))

P, A = [], []
for r in rows:
    text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    P.append(tok(text, add_special_tokens=False).input_ids)
    A.append(tok(r["response"], add_special_tokens=False).input_ids)

res = [None] * len(rows)
order = sorted(range(len(rows)), key=lambda i: len(P[i]) + len(A[i]))
batches, cur = [], []
for i in order:
    if cur and (len(cur) + 1) * (len(P[i]) + len(A[i])) > MAX_TOK:
        batches.append(cur)
        cur = []
    cur.append(i)
batches.append(cur)
with torch.no_grad():
    for bi, b in enumerate(batches):
        L = max(len(P[i]) + len(A[i]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            s = P[i] + A[i]
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        h = model.model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).last_hidden_state
        # positions len(P)-1 .. len(P)+len(A)-1 predict the answer tokens, then the end-of-turn token
        sel = [(j, t) for j, i in enumerate(b) for t in range(len(P[i]) - 1, len(P[i]) + len(A[i]))]
        logits = model.lm_head(h[[j for j, _ in sel], [t for _, t in sel]]).float()
        lse = torch.logsumexp(logits, -1)
        tl, ti = logits.topk(K, -1)
        nxt = torch.tensor([(P[i] + A[i] + [eot])[t + 1] for (j, t) in sel for i in [b[j]]], device=DEV)
        cl = logits.gather(1, nxt[:, None])[:, 0]
        lse, tl, ti, cl, nxt = (x.cpu().numpy() for x in (lse, tl, ti, cl, nxt))
        k = 0
        for j, i in enumerate(b):
            n = len(A[i])
            res[i] = (slice(k, k + n), k + n, lse, tl, ti, cl, nxt)
            k += n + 1
        for j, i in enumerate(b):
            s, e, lse_, tl_, ti_, cl_, nxt_ = res[i]
            s = slice(s.start, e + 1)  # answer tokens + the end-of-turn position (kept only for answers that ended)
            res[i] = dict(chosen=nxt_[s].astype(np.int32), chosen_logit=cl_[s], lse=lse_[s], top_ids=ti_[s].astype(np.int32),
                          top_logits=tl_[s], eot_logp=float(cl_[e] - lse_[e]),
                          is_end=np.r_[np.zeros(e - s.start, bool), True])
        if bi % 20 == 0:
            print(bi, len(batches), flush=True)

meta = [{k: r[k] for k in ("endpoint", "id", "repeat", "src")} | {"n": len(A[i]), "n_prompt": len(P[i]), "eot_logp": res[i]["eot_logp"]}
        for i, r in enumerate(rows)]
cols = {k: np.concatenate([res[i][k] for i in range(len(rows))]) for k in ("chosen", "chosen_logit", "lse", "top_ids", "top_logits", "is_end")}
cols["row"] = np.concatenate([np.full(len(A[i]) + 1, i, np.int32) for i in range(len(rows))])
np.savez_compressed(out, **cols, meta=json.dumps(meta), model=model_id)
print("wrote", out, len(cols["row"]), "tokens", flush=True)
