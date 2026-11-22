# T3 Unlearning — research notes (2026-11-21, 22h30)

Research only: nothing submitted, nothing launched on JURECA. Sources: 2023-2026 literature on "no retain data", reading of the metric, local read-only CPU diagnostics (`model.pt`, `orig_logits.pt`, 200 to 1400 forget-set images). No lead was run end to end (no local GPU): the expected effects are hypotheses.

**Top 3.** **E1** explicit targets at T=2: flip fraction `f` and margin `m` swept, wrong class taken from orig's top-5. Our last two score gaps say we still under-forget, and a third of the current flips target an implausible class. **E2** sign-consensus merge of K runs + coefficient λ: removes the drift due to Adam noise, almost free. **E3** train only the last 2-3 blocks, with SGD or with a saliency mask.

## 1. What the metric imposes (measured on orig)

- **Everything is judged at T=2.** Orig on the forget set: maxprob 0.9997 at T=1, 0.985 at T=2; top logit ≈ 23, label / 2nd class margin 12.2 on average (5 % quantile: 7.4). A diffuse tail is very expensive: 2 % of T=1 mass spread over 98 classes becomes 53 % of the mass at T=2. This is why `orig/T=4` scored 0.010 (maxprob T=2 ≈ 0.16). We must keep ≥ 10 of margin against implausible classes.
- **KL(p_gold‖p_sub) is mass-covering**, and p_gold is an average of several models. KL at T=2 of an image flipped toward class c, depending on the mass ε left on the label:

  | p_gold (c / label) | ε = 0.001 (one-hot) | ε = 0.03 | ε = 0.11 | ε = 0.25 |
  |---|---|---|---|---|
  | 0.10 / 0.90, gold was right | 5.9 | 2.8 | 1.7 | 0.95 |
  | 0.60 / 0.40, golds split | 2.1 | 0.75 | 0.28 | 0.05 |
  | 0.95 / 0.05, gold flips too | 0.15 | 0.01 | 0.02 | 0.14 |

- **mean/std** come from each gold model taken alone: heavy-tailed distribution. With a fraction `f` of flips at loss `L`: mean ≈ f·L, std ≈ L·√(f(1−f)). Leaving ε on the label at T=2 amounts to L ≈ 2·ln((1−ε)/ε): ε = 0.25 / 0.10 / 0.03 gives L ≈ 2.2 / 4.4 / 7.0. sub9_0 and sub10_2 (agreement 0.80, std 1.81) are therefore already around L ≈ 4.5 on average, because they under-learn the ×20 target, but without per-image control.
- **We still under-forget.** Every increase in forgetting has paid: crops 0.08 → 0.05 (+31 %), `w_retain` 2 → 1 (+29 %). The optimum in `f` is not bracketed.
- **Classes** (cosine of the head rows, orig's runner-ups): classes 0-59 in "sibling groups" of neighbouring indices (about 15 groups of 3 to 5), classes 60-99 mostly isolated. 85 % of the forget set is in the sibling groups, including the 8 classes at 130 images (130 looks like a per-class cap: class 6 has 129, class 32 has 104). 73 % of orig's off-label T=2 mass falls in its first 5 runner-ups. Working hypothesis: gold is most often wrong within the sibling group.
- **The crops teacher is diffuse** (200 images, 8 crops at scale 0.05-0.35): 59 % of its mass is off-label, of which only 37 % is in the top-5 of orig's clean logits. 35 % of its flips target a class outside that top-5, only 25 % the runner-up. An FGSM step (ε = 0.01 in [-1, 1] units) on the contrary flips toward the runner-up in 91 % of cases. The ranking by label mass under crops is almost uncorrelated with the clean margin (Spearman 0.12) and with the margin after FGSM (0.12).
- **Where the class is decided.** Leave-one-out kNN on orig's CLS token (1400 images, label = orig's argmax): 35 % at block 9, 54 % at block 10, 85 % at block 11, 99.6 % at block 12. Out-of-fold linear probe: 80 % / 97 % / 100 % at blocks 10 / 11 / 12. The class forms in the last 3 blocks. The `ls2` (MLP) are ≈ 0.001-0.007 on average in blocks 1-7 and 0.06-0.085 in blocks 8-12.

## 2. What the literature says

| Source | What we take from it |
|---|---|
| RUM, NeurIPS 2024 — [2406.01257](https://arxiv.org/abs/2406.01257), [code](https://github.com/kairanzhao/RUM) | Gold's accuracy on a forgotten image depends on its memorisation: ≈ 100 % (low), 53-74 % (medium), 3-23 % (high). "Doing nothing" is optimal for the low ones, random relabelling over-forgets them. Treat the forget set by subgroups. |
| TMU — [2408.11433](https://arxiv.org/abs/2408.11433) | Randomly drawn forget set: gold's accuracy on the forget set ≈ test accuracy (86.9 / 85.3 on ResNet, 84.6 / 84.6 on ViT). Their "is gold right on this image" predictor reaches ≈ 89 % for a base rate ≈ 86 %: it is hard. |
| LoTUS, CVPR 2025 — [2503.18314](https://arxiv.org/abs/2503.18314), [code](https://github.com/cspartalis/LoTUS) | Target = orig softened until the forget accuracy equals orig's on unseen data. ViT: lr 1e-6, 3 epochs. Uses 30 % of the retain set. |
| Boundary Unlearning, CVPR 2023 — [2303.11570](https://arxiv.org/abs/2303.11570) | Gold predicts "specific classes, not random ones". Shrink: one FGSM step, label = orig's output on the perturbed image; lr 1e-5, 10 epochs, forget set only. |
| SCAR, ECCV 2024 — [2404.12922](https://arxiv.org/abs/2404.12922), [code](https://github.com/jbonato1/SCAR) | No retain: distillation of orig on out-of-distribution images + attraction toward the nearest wrong class (Mahalanobis). ImageNet and COCO are equivalent, size matters little (plateau around 6 k), Gaussian noise fails (16 % accuracy). |
| NegMerge, ICML 2025 — [2410.05583](https://arxiv.org/abs/2410.05583), [code](https://github.com/naver-ai/negmerge) | Forget set only: merge K task vectors keeping only the elements with unanimous sign. Mean gap to retraining 1.07 (SalUn 1.15) on CIFAR-10. |
| NeurIPS 2023 competition — [2406.09073](https://arxiv.org/abs/2406.09073) | The top = *erase* (noise σ 0.6 on the weights, or reinitialisation of 30 to 99 % of the weights) then *repair* by cross-entropy on the retain set. All use the retain set. |
| MU-Mis — [2402.15109](https://arxiv.org/abs/2402.15109), comparison table | Random forgetting CIFAR-10 without retain, test accuracy: NegGrad 90.5, random labels 88.3, JiT 89.6, MU-Mis 91.5, SCAR 92.4, against 94.7 at the start. None matches retraining. |
| SSD / LFSSD — [2308.07707](https://arxiv.org/abs/2308.07707), [2402.19308](https://arxiv.org/abs/2402.19308) | Dampen the weights specific to the forget set (α 3.5-10, λ 1). LFSSD without labels: importance = \|∂‖f(x)‖²/∂θ\|. Random forgetting on ViT: almost no effect. |
| JiT — [2402.01401](https://arxiv.org/abs/2402.01401), [code](https://github.com/jwf40/Information-Theoretic-Unlearning) | ViT, random forgetting: η 0.01, σ 0.5. In the code: SGD without momentum, 25 noise draws, sum over the batch, noisy branch without gradient. |
| U-LiRA — [2403.01218](https://arxiv.org/abs/2403.01218) | Per-example membership attack: vulnerability drops at the start of unlearning then rises again if we continue. There is a per-example stopping point. |
| Task references — [SCRUB](https://arxiv.org/abs/2302.09880), [Zero-Shot MU](https://arxiv.org/abs/2201.05629), [SalUn](https://arxiv.org/abs/2310.12508) | SCRUB: maximises the KL to the teacher on the forget set, minimises it on the retain set, then "rewinds" until the forget error equals that of a reference model: aim for gold's level, not the maximum. Zero-Shot MU: per-class optimised noise as a retain proxy, for class forgetting. SalUn: random labels restricted to the most salient weights. All assume the retain set or whole-class forgetting. |
| Difficulty proxies — [2106.09647](https://arxiv.org/abs/2106.09647), [1910.13427](https://arxiv.org/abs/1910.13427), [1911.05248](https://arxiv.org/abs/1911.05248) | Signals related to a model that has not seen the image misclassifying it: prediction depth (per-layer kNN), adversarial distance, disagreement of a pruned model. |

## 3. Experiments, in priority order

### E1 — Explicit targets at T=2: flip fraction `f`, margin `m`, class in the top-k

- **Idea.** Keep the current smoothed teacher (16 crops) to rank the images by mass left on the label. The `f·N` weakest are flipped toward the teacher's best class taken from the top-`k` of orig's clean logits: without this restriction, a third of the flips target a class that orig considers implausible for the whole image (§1). Target = orig's logits on the augmented view (`tf`), modified for the flipped images: `alt` gets the label's logit, the label goes to `alt − m`. For the other images, `tf` unchanged: doing nothing is optimal for weakly memorised examples (RUM), and these images become real retain anchors.
- **Loss.** `loss_f = 4 * kl(t2, sf, 2.0)`: the metric itself (gold‖sub direction, T=2, tail included). The factor 4 = T² keeps the current balance with `w_retain`.
- **Flags.** `--flip_frac 0.2|0.3|0.4|0.5`, `--flip_margin 2.2|4.4|7`, `--topk 1|5|100` (1 = runner-up, 100 = no restriction). The rest as sub10_0. With `pm` = mean of the crops before `--sharpen`:

  ```python
  lo = tt.predict(orig, imgs); y0 = lo.argmax(1)                        # logits propres d'orig
  flip = torch.zeros(N, dtype=torch.bool)
  flip[pm[torch.arange(N), y0].argsort()[:int(args.flip_frac * N)]] = True   # plus faible masse sur le label
  top = lo.topk(args.topk + 1, 1).indices[:, 1:]                        # top-k hors label
  alt = top.gather(1, pm.gather(1, top).argmax(1, keepdim=True)).squeeze(1)
  # boucle : i = lignes basculées du batch, yb / ab = y0[idx] / alt[idx] sur le device
  t2 = tf.clone(); a = tf[i, yb[i]]; t2[i, ab[i]] = a; t2[i, yb[i]] = a - args.flip_margin
  loss_f = 4 * kl(t2, sf, 2.0)
  ```
- **First job, 8 lines.** `f` ∈ {0.2, 0.3, 0.4, 0.5} with `m` = 4.4 and `--topk 5`; then at `f` = 0.3: `m` = 7, `m` = 2.2, `--topk 1`, `--topk 100`. Without code, to bracket `f` right away: `--strong 0.03` and `--strong 0.02` on the sub10_0 config (read with caution: the smaller the crop, the less plausible the targeted class).
- **Altered copies (half of the forget score).** Add to the `aug` of the forget term: Gaussian noise σ 0.05, 3×3 blur (p = 0.3), rotation ±10°. In `report()`, measure agreement and loss on an alteration absent from training (downscale ×0.75 then upscale) and the T=2 mass outside the top-5.
- **Next.** Third knob, to touch only once `f` and `m` are fixed: subtract 2 or 4 from the label logit of the kept images (`--keep_drop`), in case gold is less sure than orig on an easy unseen image.
- **Expected effect.** prediction ↑ (loss = metric, hedged flips, plausible classes); mean_std tuned by `f` and `m`; retain ↑ a little. Cost: that of a current run.
- **Safeguard.** `f` ≥ 0.4 moves the boundaries within the sibling groups: monitor agreement on the kept images (gate at 70 %).

### E2 — Sign-consensus merge of K runs, then λ

- **Idea.** K runs of the same config with different seeds (teacher crops, adversarial pool, batches). τ_k = θ_k − θ_orig. Keep the elements whose sign is the same in all K runs, average: θ = θ_orig + λ · mean(τ_k) · mask. This is NegMerge applied to our forgetting vectors.
- **Why.** AdamW moves all trained weights by about lr·√steps ≈ 3 to 6e-3, even without a consistent gradient: i.e. 10 to 20 % of a weight's standard deviation (≈ 0.03), over 43 M parameters (order of magnitude computed, not measured). This random walk is pure drift (KL on the control set). The part shared by the seeds is the forgetting. NegMerge thus zeroes 77 % of the elements with 5 models and 90 % with 30. An average of runs also looks more like p_gold, an average of several models.
- **Code.** `--seed` in `unlearn.py` (instead of `torch.manual_seed(0)`) and a `merge.py`. One job = 4 seeds in parallel, then merge and `report()` for λ ∈ {1, 1.5, 2}. Control: uniform soup (mask = 1).

  ```python
  out = {}
  for k, w in orig_sd.items():
      s = torch.stack([r[k] - w for r in runs])                         # τ_k
      out[k] = w + lam * s.mean(0) * ((s > 0).all(0) | (s < 0).all(0))  # signe unanime
  ```
- **Immediate try, no GPU.** Merge the checkpoints still on disk (sub10_0, sub10_2, sub9_0). Also extrapolate the best single one: θ_orig + λ·(θ_sub10_0 − θ_orig) with λ = 1.2 and 1.5, which pushes the forgetting.
- **Choice of λ.** The one that gives back the agreement and forget std of the best single run. Submit if the KL proxy drops clearly at equal forgetting. Quick check: cos(τ_a, τ_b) per block between two seeds; below 0.3, noise dominates.
- **Expected effect.** retain ↑; forget unchanged at adjusted λ; prediction ↑ a little where the seeds diverge. Cost: one job of 4 runs, merge on CPU.
- Refs: NegMerge, [model soups](https://arxiv.org/abs/2203.05482), [WiSE-FT](https://arxiv.org/abs/2109.01903).

### E3 — Reduce drift at the source

- **Flags.** (a) `--blocks 3` and `--blocks 2`, no code: this is where the class forms (§1), 14 M parameters instead of 43 M for 2 blocks. (b) `--opt sgd --lr 0.03|0.1 --epochs 16` (the flag exists, add a gradient clip at 1.0): SGD moves only the weights with a consistent gradient. (c) `--mask 0.5`: SalUn mask, update only the 50 % of weights with the highest \|∇ loss_f\|, computed once. (d) `--train mlp`: MLP and `ls2` of the trained blocks, plus the head. (e) AdamW `--lr 1e-4 --epochs 16`.
- **Reference points.** LoTUS fine-tunes a ViT at lr 1e-6, Boundary Shrink at 1e-5, SalUn at 0.013 with mask 0.5. We are at 3e-4 with AdamW on 6 blocks.
- **Expected effect.** retain ↑; forgetting slows down, to be compensated by `--epochs` or `--w_retain 0.5`. Same or lower cost. Stacks with E1 and E2.

### E4 — Better retain proxy

- **(a) Natural out-of-distribution images**, if the organizers allow it: the PDF says "from the original model and the forget set alone", so ask first. SCAR stays within 0-4 points of the retrained model's test accuracy by distilling orig on 10 k ImageNet or COCO images, with a weight 5 to 8 times that of forgetting. Flags: `--ood_dir`, `--w_retain 4`.
- **(b) Without external data: light inversion.** The `--adv` pool does 10 PGD steps from a clean image. Move to 40 steps (step ε/10), ε 0.5 to 1.0, start = `mix(x)`, random shift ±16 px and flip at each step, 35 images per target class, pool regenerated halfway. Same objective as ViT inversion ([SMI](https://arxiv.org/abs/2510.27186): cross-entropy + total variation, 4000 iterations at 1-2 it/s) without its cost. Flags: `--adv 0.5 --adv_steps 40 --adv_mix`. About 5 min more per run.
- **(c) Stricter anchor.** Add an MSE on the logits (weight 0.1 to 1) to the T=2 KL: it also constrains the scale and the tail ([Kim et al.](https://arxiv.org/abs/2105.08919)). Flag: `--w_mse`.
- **Expected effect.** retain ↑. The adversarial pool alone already gave +15 %. See also [ZS-PAG](https://arxiv.org/abs/2507.21738).

### E5 — Which images to flip: compare the rankings

- **Observation.** The two available rankings are almost uncorrelated (§1): one of the two chooses badly. Predicting per image whether gold is right is hard anyway (TMU), hence the hedged margin of E1.
- **Flag `--rank crops|margin|fgsm|avg`**, at fixed `f` and `m`. `crops` = label mass under crops (current). `margin` = label logit − runner-up on the clean image (free). `fgsm` = margin after one FGSM step, ε 0.003 to 0.005 in [-1, 1] units (at 0.01, 75 % of the images already flip). `avg` = mean of the ranks.
- **Weak signals, do not dig.** The out-of-fold probe at block 11 misses only 3 % of the images and overlaps little with the low margins (AUC 0.67). At block 12 everything is separated at 100 %: looking for hard images in the final features (nearest centroid à la SCAR, kNN) will give nothing.
- **Not tested.** Disagreement of an orig pruned at 30-50 % by magnitude.
- **Expected effect.** prediction ↑ if a ranking beats the crops. Cost: 4 runs = one job, 4 submissions.

## 4. To avoid

- **Soft targets everywhere** (orig/T, label smoothing, global temperature on the head): at T=2 the tail explodes, and touching the head also changes the control set.
- **Gradient ascent, NegGrad, [NPO](https://arxiv.org/abs/2404.05868), random labels, uniform target** (Fanchuan's phase 1, Bad Teacher): none of them chooses the class gold would choose, and the loss exceeds gold's. NPO only slows down the divergence.
- **JiT**: our try differed from the code (momentum, mean instead of sum, a single draw), but even fixed it smooths without targeting gold's class. Do not come back to it.
- **SSD / LFSSD**: almost no effect in random forgetting, and SSD's Fisher vanishes on images learned at p = 0.9997 (gradient ∝ 1 − p).
- **NeurIPS 2023-style erase** (reset, noise on the weights): the retain set is needed to repair. The noise served their metric (distribution over 512 models); here it is drift.
- **Full DeepInversion-style inversion**: too expensive for the gain. **Gaussian noise as a proxy**: does not work (SCAR).
- **Rerunning head-bias probes**: their 0.0 score does not settle the "whole classes" hypothesis. −14.5 on 8 classes may just as well push validation accuracy below 70 %.
- **Deciding on ±5 % of public score**: it covers only 30 % of the images. Keep only clear gains along a one-parameter sweep.

## 5. Limits

- Gold's statistics (accuracy, loss, confidence) are unknown: E1 is a sweep, not a value to apply.
- CPU diagnostics on randomly drawn subsets: 1400 images for the kNN, 200 for the teachers (8 crops instead of 16). Redoing them on the 3500 on GPU takes a minute.
- The RUM, MU-Mis, NeurIPS 2023 and SMI figures come from automatic summaries of the HTML pages (appendices): orders of magnitude, to be rechecked before citing them elsewhere.
