"""Positive control for the GPTQ / AWQ replica test (QCKPT=dir, same dequantisation as lm_score.py), else null reference for collect_ep_long2.py: N plain copies of the base on prompts 8-19 of prompts_v6_long.json (T = 0.7,
no truncation, no repetition penalty), one answer each, endpoints sim-plainA.. .  Usage: python sim_plain.py BASE OUT.jsonl [N]"""
import json
import sys

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6
import os
PF, START, END, OFF = os.environ.get("PROMPTS", "prompts_v6_long.json"), int(os.environ.get("START", "8")), int(os.environ.get("END", "20")), int(os.environ.get("OFF", "0"))
P = json.load(open(PF))  # ids written = index + OFF (same ids as the ep collector)
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
DEV = "cuda"
NAME = os.environ.get("NAME", "sim-plain")
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
with open(out, "w") as fh:
    for k in range(n):
        torch.manual_seed(100 + k)
        ids = list(range(START, min(END, len(P))))
        texts = [tok.apply_chat_template([{"role": "user", "content": P[i]}], tokenize=False, add_generation_prompt=True) for i in ids]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=256,
                               repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        for i, seq in zip(ids, o[:, enc.input_ids.shape[1]:]):
            fh.write(json.dumps({"endpoint": f"{NAME}{chr(65 + k)}", "id": str(i + OFF), "repeat": 0, "prompt": P[i],
                                 "response": tok.decode(seq, skip_special_tokens=True)}) + "\n")
        print("generated", k, flush=True)
