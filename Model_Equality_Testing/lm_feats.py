"""Per-endpoint features from lm_score.py outputs (runs on CPU).

    python lm_feats.py ident scores/ep_*.npz          # which local model explains each endpoint best (log-lik / answer)
    python lm_feats.py feats scores/ref_smol2_360m.npz  # temperature fit + goodness of fit per endpoint

Goodness of fit: for a tempered reference q_T = softmax(z/T), S = sum_t [log q_T(y_t) + H(q_T,t)] has mean 0 when the
endpoint samples from q_T. z = S / sqrt(sum_t Var_q(log q)). Same weights at another temperature -> z ~ 0 after the fit;
different weights (LoRA, quantization) -> z < 0. Entropies are computed on the stored top-K logits (tail ignored).
"""
import collections
import json
import os
import sys

import numpy as np


# max_tokens of each collector; an answer this long (minus re-tokenisation slack) was cut, not ended
CAPS = {"ep_v6_long": 256, "ep_v6_long2": 256, "ep_v2_short": 16, "ep_ident": 48, "ep_v7_sysleak": 96, "ref_v6_long": 256,
        "ref_oos_long": 256, "ref_oos_ident": 48, "ref_v8_probe": 256, "ep_v8_probe": 256, "ref_v7_sysleak": 96, "sim_q15b": 256, "sim_q3b": 256, "sim_l1b": 256,
        "sim_q15bw": 256, "sim_q3bw": 256, "sim_l1bw": 256, "sim_q3bp": 256, "sim_q3bq": 256, "sim_g8": 256, "ep_v9_long": 256}


def load(path, keep=None):
    """-> (per-token columns, per-answer meta, model id). Newer files also score the end-of-turn position after each
    answer; it is kept only for answers that ended by themselves: shorter than the source's request cap (CAPS), or for
    unknown sources than the longest KEPT answer, minus one token. The cap must not come from other families' answers:
    a Llama answer re-tokenised by Qwen can reach 280 tokens and made every capped Qwen answer look ended (bug fixed
    2026-11-22). EOT=off drops every end-of-turn position (ablation)."""
    d = np.load(path)
    cols, meta = {k: d[k] for k in d.files if k not in ("meta", "model")}, json.loads(str(d["meta"]))
    if "is_end" in cols:
        cap = collections.defaultdict(int)
        for m in meta:
            if keep is None or keep(m):
                cap[m["src"]] = max(cap[m["src"]], m["n"])
        lim = {s: CAPS.get(s.split(".")[0], c) for s, c in cap.items()}
        off = os.environ.get("EOT") == "off"
        ended = np.array([not off and m["src"] in lim and m["n"] < lim[m["src"]] - 1 for m in meta])
        end_ok = ~cols["is_end"] | ended[cols["row"]]
        cols = {k: v[end_ok] for k, v in cols.items()}
    if keep is not None:  # keep(meta_row) -> bool: restrict to a subset of answers (stability checks)
        ok = np.array([bool(keep(m)) for m in meta])
        cols = {k: v[ok[cols["row"]]] for k, v in cols.items()}
    return cols, meta, str(d["model"])


def ident(paths):
    res = {}
    for p in paths:
        cols, meta, model = load(p)
        lp = cols["chosen_logit"] - cols["lse"]
        per_row = np.bincount(cols["row"], weights=lp, minlength=len(meta))
        by = collections.defaultdict(list)
        for m, v in zip(meta, per_row):
            by[m["endpoint"]].append(v)
        res[model.split("/")[-1]] = {e: float(np.mean(v)) for e, v in by.items()}
    models = list(res)
    eps = sorted(res[models[0]])
    print("mean log-lik per answer (higher = better explained)\n" + " " * 8 + " ".join(f"{m[:14]:>14}" for m in models))
    for e in eps:
        vals = [res[m][e] for m in models]
        best = int(np.argmax(vals))
        print(f"{e:8s}" + " ".join(("*" if i == best else " ") + f"{v:13.1f}" for i, v in enumerate(vals)))


TGRID = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0])


def tempered(cols, T):
    """log q_T(chosen), entropy and variance of log q_T per token, with q_T = softmax(z/T) restricted to the stored
    top-K (a proper conditional likelihood: callers keep only tokens whose chosen id is inside the top-K)."""
    tl = cols["top_logits"].astype(np.float64) / T
    ch = cols["chosen_logit"].astype(np.float64) / T
    lse = np.logaddexp.reduce(tl, 1)
    lq = tl - lse[:, None]
    q = np.exp(lq)
    H = -(q * lq).sum(1)
    V = (q * lq ** 2).sum(1) - H ** 2
    return ch - lse, H, V


def feats(path, min_tokens=1):
    cols, meta, model = load(path)
    ep_of_row = np.array([m["endpoint"] for m in meta])
    ep_tok = ep_of_row[cols["row"]]
    rank1 = cols["top_ids"][:, 0] == cols["chosen"]
    p1 = np.exp(cols["top_logits"][:, 0] - cols["lse"])
    stats = {T: tempered(cols, T) for T in TGRID}
    print(f"model {model}\n{'ep':8s} {'tok':>6s} {'T*':>5s} {'z(T*)':>7s} {'z(0.7)':>7s} {'z(1.0)':>7s} "
          f"{'top1':>5s} {'top1|p1>.9':>10s} {'top1|.5-.9':>10s}")
    for e in sorted(set(ep_of_row)):
        s = ep_tok == e
        ll = {T: stats[T][0][s].sum() for T in TGRID}
        Ts = max(TGRID, key=lambda T: ll[T])
        z = {T: (stats[T][0][s] + stats[T][1][s]).sum() / np.sqrt(stats[T][2][s].sum()) for T in TGRID}
        hi, mid = s & (p1 > 0.9), s & (p1 > 0.5) & (p1 <= 0.9)
        print(f"{e:8s} {s.sum():6d} {Ts:5.1f} {z[Ts]:7.2f} {z[0.7]:7.2f} {z[1.0]:7.2f} {rank1[s].mean():5.2f} "
              f"{rank1[hi].mean():10.3f} {rank1[mid].mean():10.3f}")


def llr(path0, path1, T=0.7):
    """Per-endpoint log-likelihood ratio log p1(y) - log p0(y) of the answers (e.g. p1 = local fp8 replica, p0 = bf16),
    at T=1 (exact) and at temperature T (top-K normalizers). Same rows in both files (same answers scored twice)."""
    c0, meta, _ = load(path0)
    c1, meta1, _ = load(path1)
    assert len(meta) == len(meta1) and (c0["chosen"] == c1["chosen"]).all()
    l1 = (c1["chosen_logit"] - c1["lse"]) - (c0["chosen_logit"] - c0["lse"])
    lT = tempered(c1, T)[0] - tempered(c0, T)[0]
    ep_tok = np.array([m["endpoint"] for m in meta])[c0["row"]]
    print(f"{'ep':8s} {'tok':>6s} {'LLR@1 sum':>10s} {'z':>6s} {'LLR@T sum':>10s} {'z':>6s}")
    for e in sorted(set(ep_tok)):
        s = ep_tok == e
        z1 = l1[s].sum() / (l1[s].std() * np.sqrt(s.sum()))
        zT = lT[s].sum() / (lT[s].std() * np.sqrt(s.sum()))
        print(f"{e:8s} {s.sum():6d} {l1[s].sum():10.2f} {z1:6.2f} {lT[s].sum():10.2f} {zT:6.2f}")


def prefix(paths):
    """Disguise signature from repeated prompts (RESEARCH.md idea 1): mean common-prefix length (chars) between answers
    to the same prompt, exact-duplicate rate, first-word agreement, mean length. Watermark -> long prefixes."""
    by = collections.defaultdict(list)
    for f in paths:
        for l in open(f):
            r = json.loads(l)
            by[(r["endpoint"], r["id"])].append(r["response"].strip())
    eps = sorted({e for e, _ in by})
    print(f"{'ep':8s} {'pairs':>5s} {'prefix':>7s} {'dup':>5s} {'word1':>6s} {'len':>6s}")
    for e in eps:
        pre, dup, w1, ln = [], [], [], []
        for (e2, _), xs in by.items():
            if e2 != e:
                continue
            ln += [len(x) for x in xs]
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    a, b = xs[i], xs[j]
                    k = next((n for n, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
                    pre.append(k)
                    dup.append(a == b)
                    w1.append(a.split()[:1] == b.split()[:1])
        if pre:
            print(f"{e:8s} {len(pre):5d} {np.mean(pre):7.1f} {np.mean(dup):5.2f} {np.mean(w1):6.2f} {np.mean(ln):6.0f}")


def table(path, fp8=None, eps=None, T_grid=np.arange(0.4, 1.61, 0.025), keep=None):
    """One row per endpoint: fitted temperature T^, NLL, fp8-replica likelihood ratio z (if given), and per source file
    the top-1 agreement z on confident tokens (observed minus expected under T^; end-of-turn decisions included).
    On ref-*: LoRA -> top-1 z << 0 on the one-word prompts (it does not stop where the base stops); fp8 -> fp8_z >> 0;
    system prompt, persona, watermark and temperature stay near 0 on both. Returns {endpoint: feature dict}."""
    if eps and keep is None:  # family-only caps for the end-of-turn mask
        keep = lambda m: m["endpoint"] in eps
    elif eps:
        keep = (lambda k: lambda m: m["endpoint"] in eps and k(m))(keep)
    cols, meta, model = load(path, keep)
    K = cols["top_ids"].shape[1]
    ep_tok = np.array([m["endpoint"] for m in meta])[cols["row"]]
    src_tok = np.array([m.get("src", "all").split(".")[0] for m in meta])[cols["row"]]
    srcs = sorted(set(src_tok))
    hit = cols["top_ids"] == cols["chosen"][:, None]
    rank = np.where(hit.any(1), hit.argmax(1), 99)
    p1 = np.exp(cols["top_logits"][:, 0] - cols["lse"])
    if fp8:
        c1, _, _ = load(fp8, keep)
        assert (c1["chosen"] == cols["chosen"]).all()
        lr1 = (c1["chosen_logit"] - c1["lse"]) - (cols["chosen_logit"] - cols["lse"])
    print(f"model {model}\n{'ep':8s} {'tok':>6s} {'T^':>6s} {'nll@1':>6s}" + (f" {'fp8_z':>6s}" if fp8 else "")
          + "".join(f" {s[-14:]:>14s}" for s in srcs))
    out = {}
    for e in eps or sorted(set(ep_tok)):
        s = (ep_tok == e) & (rank < K)  # conditional on the chosen token being in the top-K
        T = max(T_grid, key=lambda T: tempered({k: v[s] for k, v in cols.items()}, T)[0].sum())
        tl = cols["top_logits"].astype(np.float64) / T
        q1 = np.exp(tl[:, 0] - np.logaddexp.reduce(tl, 1))
        f = {"T": T, "nll": float(-(cols["chosen_logit"][s] - cols["lse"][s]).mean())}
        if fp8:
            x = lr1[s]
            f["fp8_z"] = float(x.sum() / (x.std() * np.sqrt(len(x))))
        for src in srcs:
            c = s & (src_tok == src) & (p1 > 0.6)
            f[src] = float(((rank[c] == 0).sum() - q1[c].sum()) / np.sqrt((q1[c] * (1 - q1[c])).sum() + 1e-9))
        out[e] = f
        print(f"{e:8s} {s.sum():6d} {T:6.3f} {f['nll']:6.3f}" + (f" {f['fp8_z']:6.2f}" if fp8 else "")
              + "".join(f" {f[src]:14.2f}" for src in srcs))
    return out


DECODERS = ([("T", 0)] + [("rep", v) for v in (1.02, 1.05, 1.1, 1.2, 1.3)] + [("pres", v) for v in (0.1, 0.3, 0.6, 1.0)]
            + [("freq", v) for v in (0.1, 0.3, 0.6, 1.0)] + [("minp", v) for v in (0.02, 0.05, 0.1)])


def decoder(path, model_id, files, eps=None, T_grid=np.arange(0.5, 1.31, 0.05)):
    """Plan step 1b: does a penalty or min-p explain an endpoint better than temperature alone?  For each decoder knob,
    T is refitted; the knob is chosen on one half of the prompts (ids 0-3 vs 4-7 of the long source, crc parity
    otherwise) and its gain over T-only is measured on the other half, then the halves swap.  vLLM order: penalties on
    the logits (repetition on prompt + answer history, sign-aware; presence / frequency on the answer only), then T,
    then min-p.  Same stored top-K support for every knob.  Positive control: the Qwen simulators inherit
    repetition_penalty 1.1 (1.5B) / 1.05 (3B), the fit must find it.
    Usage: python lm_feats.py decoder SCORES.npz MODEL answers1.jsonl [...]   (same files, same order as lm_score.py)"""
    import zlib
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    cols, meta, _ = load(path, (lambda m: m["endpoint"] in eps) if eps else None)
    assert len(rows) == len(meta) and all(r["endpoint"] == m["endpoint"] for r, m in zip(rows, meta)), "files != archive"
    K = cols["top_ids"].shape[1]
    hit = cols["top_ids"] == cols["chosen"][:, None]
    rank = np.where(hit.any(1), hit.argmax(1), K)
    inhist, cnt = np.zeros(hit.shape, bool), np.zeros(hit.shape, np.float32)
    for r in np.unique(cols["row"]):  # token history of every answer, rebuilt before any position filter
        idx = np.flatnonzero(cols["row"] == r)
        text = tok.apply_chat_template([{"role": "user", "content": rows[r]["prompt"]}], tokenize=False,
                                       add_generation_prompt=True)
        seen, c = set(tok(text, add_special_tokens=False).input_ids), collections.Counter()
        for i in idx:
            ids = cols["top_ids"][i]
            cnt[i] = [c[t] for t in ids]
            inhist[i] = [t in seen or c[t] > 0 for t in ids]
            c[int(cols["chosen"][i])] += 1
    z0 = cols["top_logits"].astype(np.float64)

    def ll(sel, knob, v, T):
        z = z0[sel]
        if knob == "rep":
            z = np.where(inhist[sel], np.where(z > 0, z / v, z * v), z)
        elif knob == "pres":
            z = z - v * (cnt[sel] > 0)
        elif knob == "freq":
            z = z - v * cnt[sel]
        l = z / T
        q = np.exp(l - np.logaddexp.reduce(l, 1, keepdims=True))
        if knob == "minp":  # only where the cutoff falls inside the stored top-K
            cut = q >= v * q.max(1, keepdims=True)
            q = np.where(cut | (q[:, -1:] >= v * q.max(1, keepdims=True)), q, 0)
            q /= q.sum(1, keepdims=True)
        r = rank[sel]
        qy = np.where(r < K, q[np.arange(len(r)), np.minimum(r, K - 1)], 0)
        return np.log(np.maximum(qy, 1e-5)).sum()

    ep_tok = np.array([m["endpoint"] for m in meta])[cols["row"]]
    fold_row = np.array([(int(m["id"]) >= 4) if m["src"].startswith(("ep_v6_long", "sim_", "ref_oos_long", "ref_v6_long"))
                         and m["id"].isdigit() else zlib.crc32(f"{m['src']}|{m['id']}".encode()) % 2 for m in meta])
    fold = fold_row[cols["row"]]
    print(f"{'ep':12s} {'held-out gain vs T-only (nats)':>32s}   knob chosen on each half")
    for e in eps or sorted(set(ep_tok)):
        gain, picks = 0.0, []
        for train in (0, 1):
            tr, te = (ep_tok == e) & (fold == train), (ep_tok == e) & (fold != train)
            best = max(DECODERS, key=lambda d: max(ll(tr, *d, T) for T in T_grid))
            T_b = max(T_grid, key=lambda T: ll(tr, *best, T))  # T refitted with the knob on the training half
            gain += ll(te, *best, T_b) - ll(te, "T", 0, max(T_grid, key=lambda T: ll(tr, "T", 0, T)))
            picks.append(f"{best[0]}={best[1]}")
        print(f"{e:12s} {gain:32.1f}   {' / '.join(picks)}")


def axis(path, model_id, anchors, eps=None, temps=None, n_folds=8, lam=1.0):
    """Plan step 2: blind, cross-fitted LoRA-direction test.  Per answer position, a small fixed feature vector of the
    chosen token (end of turn, '\\n\\n', '**', '#', list number, digit, capitalised, rank bucket under the base 1 / 2 /
    3-5 / 6-20 / 21-64), each split into 'first 8 tokens' and 'later', minus its expectation under the base at the
    endpoint's temperature (stored top-K).  The direction is fitted on an anchor endpoint (a probable LoRA) on the
    training folds and every endpoint, the anchor included, is scored on the held-out fold (whole prompt ids), so no
    endpoint is scored with a direction from its own tokens:  U = sum(w . r) / sqrt(sum(w^2 . var)).  Base-model
    endpoints should score ~0; the same adapter at another strength scores > 0.
    Usage: python lm_feats.py axis SCORES.npz MODEL ANCHOR[,ANCHOR...]   (EPS= family, TEMPS= json {ep: T})"""
    import re
    import zlib
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    cols, meta, _ = load(path, (lambda m: m["endpoint"] in eps) if eps else None)
    K = cols["top_ids"].shape[1]
    eot_ids = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>") if t in tok.get_vocab()}
    tests = (lambda s: "\n\n" in s, lambda s: "**" in s, lambda s: "#" in s, lambda s: re.fullmatch(r"\s*\d+\.?", s) is not None,
             lambda s: any(c.isdigit() for c in s), lambda s: s.strip()[:1].isupper())
    vocab = np.unique(np.concatenate([cols["top_ids"].ravel(), cols["chosen"]]))
    tf = np.zeros((int(vocab.max()) + 1, 1 + len(tests)), np.float32)  # token -> class features
    for i in vocab:
        s = tok.decode([int(i)])
        tf[i] = [i in eot_ids] + [bool(t(s)) for t in tests]
    hit = cols["top_ids"] == cols["chosen"][:, None]
    rank = np.where(hit.any(1), hit.argmax(1), K)
    buckets = np.searchsorted([1, 2, 5, 20, K], np.arange(K), side="right")  # rank 0 -> 0, 1 -> 1, 2-4 -> 2, ...
    pos = np.concatenate([[0], np.flatnonzero(np.diff(cols["row"])) + 1])
    first = np.zeros(len(rank), bool)
    for s in pos:
        first[s:s + 8] = True
    ep_tok = np.array([m["endpoint"] for m in meta])[cols["row"]]
    fold_row = np.array([int(m["id"]) % n_folds if m["id"].isdigit() and m["src"].startswith(("ep_v6_long", "sim_", "ref_oos_long", "ref_v6_long"))
                         else zlib.crc32(f"{m['src']}|{m['id']}".encode()) % n_folds for m in meta])
    fold = fold_row[cols["row"]]
    R, V = {}, {}  # per endpoint and fold: summed residual vector and variance
    for e in eps or sorted(set(ep_tok)):
        s = (ep_tok == e) & (rank < K)  # conditional on the top-K, like tempered(): the expectation is normalised there
        T = (temps or {}).get(e, 0.7)
        l = cols["top_logits"][s].astype(np.float64) / T
        q = np.exp(l - np.logaddexp.reduce(l, 1, keepdims=True))
        cls = tf[cols["top_ids"][s]]                                   # [n, K, C]
        ex_c = (q[..., None] * cls).sum(1)  # binary features: E[f^2] = E[f], so var = E[f](1 - E[f])
        ob_c = tf[cols["chosen"][s]]
        ob_b = np.eye(5)[buckets[rank[s]]]
        ex_b = np.stack([q[:, buckets == b].sum(1) for b in range(5)], 1)
        ob, ex = np.hstack([ob_c, ob_b]), np.hstack([ex_c, ex_b])
        var = ex * (1 - ex)
        early = first[s][:, None]
        r = np.hstack([(ob - ex) * early, (ob - ex) * ~early])
        v = np.hstack([var * early, var * ~early])
        for k in range(n_folds):
            m = fold[s] == k
            R[e, k], V[e, k] = r[m].sum(0), v[m].sum(0)
    print(f"U along the direction of each anchor (held-out folds); |U| > ~2.5 = shares the anchor's adapter direction")
    print(f"{'ep':12s} " + " ".join(f"{a:>10s}" for a in anchors))
    for e in eps or sorted(set(ep_tok)):
        out = []
        for a in anchors:
            num = den = 0.0
            for k in range(n_folds):
                Ra = sum(R[a, j] for j in range(n_folds) if j != k)
                Va = sum(V[a, j] for j in range(n_folds) if j != k)
                w = Ra / (Va + lam)  # per-feature z-like weights; diagonal ridge
                num += w @ R[e, k]
                den += (w ** 2) @ V[e, k]
            out.append(num / np.sqrt(den + 1e-12))
        print(f"{e:12s} " + " ".join(f"{u:+10.2f}" for u in out))


def residuals(path, eps=None, T_grid=np.arange(0.4, 1.61, 0.025), srcs=None):
    """Per endpoint, a vocabulary residual r[v] = (observed - expected count of token v) / sqrt(expected + 1), the
    expectation taken under the base at the endpoint's fitted temperature (top-K conditional). LoRAs of one adapter
    share a direction (RESEARCH.md: cosine 0.44 between LoRA 0.7 and 1.5 on ref-*). Returns ({ep: r}, {ep: T})."""
    cols, meta, _ = load(path, (lambda m: m["endpoint"] in eps) if eps else None)
    K = cols["top_ids"].shape[1]
    ep_tok = np.array([m["endpoint"] for m in meta])[cols["row"]]
    src_tok = np.array([m.get("src", "all").split(".")[0] for m in meta])[cols["row"]]
    hit = cols["top_ids"] == cols["chosen"][:, None]
    rank = np.where(hit.any(1), hit.argmax(1), 99)
    V = int(max(cols["top_ids"].max(), cols["chosen"].max())) + 1
    R, Ts = {}, {}
    for e in eps or sorted(set(ep_tok)):
        s = (ep_tok == e) & (rank < K) & (np.isin(src_tok, srcs) if srcs else True)
        c = {k: v[s] for k, v in cols.items()}
        T = max(T_grid, key=lambda T: tempered(c, T)[0].sum())
        tl = c["top_logits"].astype(np.float64) / T
        q = np.exp(tl - np.logaddexp.reduce(tl, 1, keepdims=True))
        exp_ = np.bincount(c["top_ids"].ravel(), weights=q.ravel(), minlength=V)
        obs = np.bincount(c["chosen"], minlength=V)
        R[e], Ts[e] = (obs - exp_) / np.sqrt(exp_ + 1), T
    return R, Ts


def resid_report(path, eps=None, srcs=None):
    R, Ts = residuals(path, eps, srcs=srcs)
    eps = sorted(R)
    print("residual norm and cosine matrix\n" + " " * 16 + " ".join(e[-2:].rjust(5) for e in eps))
    for a in eps:
        na = np.linalg.norm(R[a])
        print(f"{a} T{Ts[a]:.2f} {na:5.1f} " + " ".join(
            "   --" if a == b else f"{R[a] @ R[b] / (na * np.linalg.norm(R[b])):5.2f}" for b in eps))


if __name__ == "__main__":
    if sys.argv[1] == "resid":
        resid_report(sys.argv[2], srcs=sys.argv[3].split(",") if len(sys.argv) > 3 else None)
        sys.exit()
    if sys.argv[1] == "axis":  # EPS=family endpoints, TEMPS=json {endpoint: fitted T}
        axis(sys.argv[2], sys.argv[3], sys.argv[4].split(","), os.environ["EPS"].split(",") if os.environ.get("EPS") else None,
             json.load(open(os.environ["TEMPS"])) if os.environ.get("TEMPS") else None)
        sys.exit()
    if sys.argv[1] == "decoder":  # EPS=ep-03,ep-06,... restricts to one family (family-only end-of-turn caps)
        decoder(sys.argv[2], sys.argv[3], sys.argv[4:], os.environ["EPS"].split(",") if os.environ.get("EPS") else None)
        sys.exit()
    {"ident": lambda: ident(sys.argv[2:]), "feats": lambda: feats(sys.argv[2]),
     "llr": lambda: llr(sys.argv[2], sys.argv[3]), "prefix": lambda: prefix(sys.argv[2:]),
     "table": lambda: table(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)}[sys.argv[1]]()
