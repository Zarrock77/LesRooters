"""Fine-tune the synthetic-pretrained detector on official train images, one frozen outer fold at a time.

Uses the team's existing nested protocol (reference/protocol.json + fold_k/inner_split.json):
  fit (324)  -> gradient updates, mixed with fresh synthetic compositions so the watermark cue is not forgotten
  tune (84)  -> checkpoint selection only
  outer (102)-> scored once with the selected checkpoint; never used for any decision
The selected checkpoint also predicts the 1020 test images (averaged over folds later).
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split, run  # noqa: E402
from train_det import (METHODS, Real, Synthetic, build_from_checkpoint, down128, loss_fn, make_mask, miou_batch,  # noqa: E402
                       sample_area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--frame-json", required=True, type=Path)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--real", type=int, default=16, help="real images per batch; the rest is synthetic")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="0 = the original stream; other values give an independent run for averaging")
    ap.add_argument("--methods", default="", help="fine-tune and select on these methods only (subset of the checkpoint's methods)")
    ap.add_argument("--all-inner", action="store_true",
                    help="train on fit + tune (26 %% more official pictures) for exactly --steps steps, no checkpoint selection; "
                         "the step count must come from earlier runs that selected on tune, never from the outer fold")
    ap.add_argument("--objects", type=float, default=0.0, help="share of synthetic transplants shaped like real objects")
    ap.add_argument("--pure", type=float, default=0.0, help="share of synthetic samples whose unwatermarked part is the same clean picture")
    ap.add_argument("--fields", type=float, default=0.0, help="share of generated synthetic holes that are blob fields")
    ap.add_argument("--shapes", type=float, default=0.0,
                    help="share of synthetic holes / object regions shaped like the official masks of this fold's training pictures")
    ap.add_argument("--teacher", type=Path, help="run whose fold-k models label the unlabelled official pictures (test + this outer fold)")
    ap.add_argument("--pseudo", type=int, default=8, help="unlabelled official pictures per batch when --teacher is given")
    ap.add_argument("--pseudo-conf", type=float, default=0.85, help="cells the teacher is less sure about are ignored")
    ap.add_argument("--cutmix", type=float, default=0.0, help="probability of recombining a real fit image with another fit image of the same method")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)        # several fine-tunes share the machine; unbounded CPU threads starve each other
    torch.manual_seed(20261121 + args.fold + 1000 * args.seed)
    rng = np.random.default_rng(20261121 + args.fold + 1000 * args.seed)

    frame = json.loads(args.frame_json.read_text())
    flat = args.out / "train_frame.json"
    flat.write_text(json.dumps(frame["train"]))
    real = Real(args.kit, flat)
    name_to_index = {n: i for i, n in enumerate(real.names)}
    protocol = json.loads((args.reference / "protocol.json").read_text())
    inner = json.loads((args.reference / f"fold_{args.fold}" / "inner_split.json").read_text())
    outer_names = protocol["folds"][args.fold]
    assert set(inner["validation"]) == set(outer_names)
    assert not (set(inner["fit"]) | set(inner["tune"])) & set(outer_names)
    state = torch.load(args.init, map_location="cuda")
    cfg = state.get("args", {})
    # A specialist checkpoint (trained on some methods only) is fine-tuned and selected on those methods only.
    used = [METHODS.index(m) for m in cfg.get("methods", ",".join(METHODS)).split(",")]
    if args.methods:           # single-method fine-tuning of a multi-method checkpoint
        assert set(args.methods.split(",")) <= {METHODS[k] for k in used}
        used = [METHODS.index(m) for m in args.methods.split(",")]
    keep = lambda names: np.array([name_to_index[n] for n in names if int(real.method[name_to_index[n]]) in used])  # noqa: E731
    fit, tune = keep(inner["fit"]), keep(inner["tune"])
    if args.all_inner:
        fit = np.concatenate([fit, tune])        # tune is training data now; its score below is information only
    outer = np.array([name_to_index[n] for n in outer_names])
    masks = torch.from_numpy(real.masks)
    fit_by_method = {m: fit[real.method[fit].numpy() == m] for m in used}

    # Unlabelled official pictures (transductive): the test images and this fold's outer images, labelled by the
    # teacher's fold-k models only, which never saw an outer-k label. Outer labels are not read here.
    pool = None
    if args.teacher:
        test_rows, test_images = load_split(args.kit, "test")
        with np.load(args.teacher / f"fold_{args.fold}" / "test_probs.npz") as z:
            assert z["names"].tolist() == [r["image_name"] for r in test_rows]
            test_probs = z["probs"].astype(np.float32)
        with np.load(args.teacher / f"fold_{args.fold}" / "outer_probs.npz") as z:
            assert z["names"].tolist() == outer_names
            outer_probs_teacher = z["probs"].astype(np.float32)
        t_method = np.array([METHODS.index(r["method"]) for r in test_rows])
        t_flag = np.array([int(frame["test"][r["image_name"]]["acc"] >= 0.8) for r in test_rows])
        keep_t = np.isin(t_method, used)
        keep_o = np.isin(real.method[outer].numpy(), used)
        pool = {"images": torch.cat([test_images[torch.from_numpy(keep_t)], real.images[outer[keep_o]]]),
                "probs": torch.from_numpy(np.concatenate([test_probs[keep_t], outer_probs_teacher[keep_o]])),
                "method": torch.from_numpy(np.concatenate([t_method[keep_t], real.method[outer].numpy()[keep_o]])),
                "flag": torch.from_numpy(np.concatenate([t_flag[keep_t], real.flag[outer].numpy()[keep_o]]))}
        pool["by_method"] = {m: np.nonzero(pool["method"].numpy() == m)[0] for m in used}
        print("PSEUDO pool", len(pool["images"]), "pictures from", args.teacher.name, flush=True)

    torch.backends.cudnn.benchmark = not cfg.get("wm_branch", False)
    model = build_from_checkpoint(state, args.kit).cuda().to(memory_format=torch.channels_last)
    shapes = None
    if args.shapes > 0:
        # Pixel masks of the official pictures this fold trains on (fit, plus tune with --all-inner), all three methods:
        # the shape of an edit does not depend on the watermark method. Outer pictures are never read.
        from PIL import Image
        holes, objects = [], []
        for n in list(inner["fit"]) + (list(inner["tune"]) if args.all_inner else []):
            m = np.array(Image.open(args.kit / "release/train/masks" / f"{n}.png").convert("L")) > 127
            assert m.shape == (512, 512)
            if frame["train"][n]["acc"] >= 0.8:
                if (~m).mean() > 0.001:
                    holes.append(~m)
            elif m.any():
                ys, xs = np.where(m)
                if m.sum() < 0.98 * (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()):      # not a filled rectangle
                    objects.append(m)
        shapes = {"hole": np.stack(holes), "object": np.stack(objects)}
        print("SHAPES official holes", len(holes), "object regions", len(objects), flush=True)
    synthetic = Synthetic(args.layers, True, args.steps * (args.batch - args.real), methods=used, objects=args.objects, pure=args.pure,
                          shapes=shapes, shape_share=args.shapes, fields=args.fields)
    loader = iter(DataLoader(synthetic, batch_size=args.batch - args.real, num_workers=args.workers, drop_last=True,
                             pin_memory=True, worker_init_fn=lambda w: cv2.setNumThreads(1)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 50) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.steps)))))
    scaler = torch.amp.GradScaler("cuda")

    def score(indices):
        model.eval()
        probs = run(model, real.images[indices], real.method[indices], real.flag[indices]).astype(np.float32)
        return probs, miou_batch(probs > 0.5, real.masks[indices].astype(bool))

    history = []
    _, s = score(tune)
    best = {"step": 0, "tune": float(s.mean()), "state": copy.deepcopy(model.state_dict())}
    _, s0 = score(outer)   # reported only: the pretrained model has never seen any official image
    history.append({"step": 0, "tune": best["tune"], "outer_before_finetune": float(s0.mean())})
    print("FT", args.fold, "step 0 tune", round(best["tune"], 4), "outer(before, info only)", round(float(s0.mean()), 4), flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        image, mask, method, _, flag = next(loader)
        pick = torch.from_numpy(rng.choice(fit, args.real, replace=False))
        real_image, real_mask, real_flag = real.images[pick].clone(), masks[pick].clone(), real.flag[pick].clone()
        if args.cutmix > 0:
            # Recombine FIT images of the same method: a random blob of image B replaces the same place in image A,
            # labels follow the pixels. Real watermark, real edited content, new shapes. Tune/outer images never enter.
            for j in range(len(pick)):
                if rng.random() >= args.cutmix:
                    continue
                a = int(pick[j])
                b = int(rng.choice(fit_by_method[int(real.method[a])]))
                if b == a:
                    continue
                blob = make_mask(rng, sample_area(rng, 0.01, 0.6, 0.8))
                blob_t, blob128 = torch.from_numpy(blob), torch.from_numpy(down128(blob))
                real_image[j][blob_t] = real.images[b][blob_t]
                real_mask[j][blob128] = masks[b][blob128]
                from_full = torch.where(blob128, real.flag[b].bool(), real.flag[a].bool())
                real_flag[j] = int((real_mask[j] & from_full).float().mean() >= 0.35)
        image = torch.cat([image, real_image])
        mask = torch.cat([mask, real_mask])
        method = torch.cat([method, real.method[pick]])
        flag = torch.cat([flag, real_flag])
        labelled = len(image)
        if pool is not None:
            # Unlabelled official pictures with the teacher's confident cells as targets; half of them recombined
            # (a blob of picture B in picture A, same method) so that official watermark and official clean content meet.
            ids = rng.choice(len(pool["images"]), args.pseudo, replace=False)
            p_image, p_prob, p_flag = pool["images"][ids].clone(), pool["probs"][ids].clone(), pool["flag"][ids].clone()
            for j, a in enumerate(ids):
                if rng.random() >= 0.5:
                    continue
                b = int(rng.choice(pool["by_method"][int(pool["method"][a])]))
                if b == int(a):
                    continue
                blob = make_mask(rng, sample_area(rng, 0.01, 0.6, 0.8))
                blob_t, blob128 = torch.from_numpy(blob), torch.from_numpy(down128(blob))
                p_image[j][blob_t] = pool["images"][b][blob_t]
                p_prob[j][blob128] = pool["probs"][b][blob128]
                from_full = torch.where(blob128, pool["flag"][b].bool(), pool["flag"][a].bool())
                p_flag[j] = int(((p_prob[j] > 0.5) & from_full).float().mean() >= 0.35)
            image = torch.cat([image, p_image])
            method = torch.cat([method, pool["method"][ids]])
            flag = torch.cat([flag, p_flag])
        image, mask, method, flag = (v.cuda(non_blocking=True) for v in (image, mask, method, flag))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(image, method, flag)
        loss = loss_fn(logits[:labelled].float(), mask)
        if pool is not None:
            p_prob = p_prob.cuda(non_blocking=True)
            weight = ((p_prob > args.pseudo_conf) | (p_prob < 1 - args.pseudo_conf)).float()
            bce = nn.functional.binary_cross_entropy_with_logits(logits[labelled:].float(), (p_prob > 0.5).float(), reduction="none")
            loss = loss + 0.5 * (bce * weight).sum() / weight.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            _, s = score(tune)
            history.append({"step": step, "tune": float(s.mean()), "loss": float(loss)})
            print("FT", args.fold, "step", step, "tune", round(float(s.mean()), 4), "loss", round(float(loss), 4),
                  round(time.time() - t0, 1), "s", flush=True)
            if args.all_inner:
                best = {"step": step, "tune": float(s.mean()), "state": copy.deepcopy(model.state_dict()) if step == args.steps else None}
            elif s.mean() > best["tune"]:
                best = {"step": step, "tune": float(s.mean()), "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    outer_probs, outer_scores = score(outer)
    result = {"fold": args.fold, "methods_trained": [METHODS[k] for k in used], "selected_step": best["step"], "tune_miou": best["tune"],
              "outer_miou": float(outer_scores.mean()), "outer_before_finetune": float(s0.mean()), "history": history}
    for mi, m in enumerate(METHODS):
        sel = real.method[outer].numpy() == mi
        result[m] = float(outer_scores[sel].mean())
    for g in ("full", "rect", "transplant"):
        sel = np.array([real.groups[i] == g for i in outer])
        result[g] = float(outer_scores[sel].mean())
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(args.out / "outer_probs.npz", names=np.array(outer_names), probs=outer_probs.astype(np.float16),
                        scores=outer_scores)
    del pool
    rows, images = load_split(args.kit, "test")
    methods = torch.tensor([METHODS.index(r["method"]) for r in rows])
    flags = torch.tensor([int(frame["test"][r["image_name"]]["acc"] >= 0.8) for r in rows])
    np.savez_compressed(args.out / "test_probs.npz", names=np.array([r["image_name"] for r in rows]),
                        probs=run(model, images, methods, flags))
    torch.save({"ema": best["state"], "args": cfg, "fold": args.fold, "step": best["step"]}, args.out / "best.pt")
    print("FT_DONE", json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
