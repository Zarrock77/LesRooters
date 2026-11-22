# T3 Unlearning — status (2026-11-22, 1h)

## Handover (for Aymeric) — read this first
- **Final score (close at 12h00): 0.4275 public** (`m28_wr10`), T3 rank ~4th tied, team 5th overall (0.93) at 11h57. Lessons from the morning, in order of gain: (1) the gold loss stats were far above our estimate (sum 6.5 → actually std ≈ 4.5-4.6 and mean ≈ 4.6-4.8): the score was 0.0486 × (mean + std) as long as we were below it (0.30 → 0.385 by raising `--m` from 8 to 11); (2) once the std was reached, raise the mean by flipping more images (33 % → 55 %, plateau 45-60 %): 0.41; (3) `--alt 0.25` aligns the loss of blurred / downscaled copies with the clean images: 0.42; (4) retain 8-10 and lr 2e-4 × 12 epochs: +0.01. Refuted: tail floor, smooth allocation M2, reordering the group, depth, `--heal`, repair, seed soup.
- **Best public score: 0.4275** (`m28_wr10`, 2026-11-22 11h47: `--flip 0.55 --hedge 0.45 --m 11.3 --alt 0.25 --w_retain 10 --hd -0.15 --lr 2e-4 --epochs 12`); before that 0.426 (`m27_alt25_m11`, 11h25: `--m 11.0 --w_retain 8`); before that 0.4212 (`m26_f55_alt`, 11h14: `--flip 0.55 --hedge 0.45 --m 10.5 --alt 0.25 --w_retain 8 --hd -0.15 --lr 2e-4 --epochs 12`); before that 0.41 (`m24_f0.55_m10.5`, 10h51, without `--alt`); before that 0.4031 (`m23_f0.45_m10.5`, 10h38: `--flip 0.45 --hedge 0.55 --m 10.5 --w_retain 8 --hd -0.15 --lr 2e-4 --epochs 12`); before that 0.3855 (`m20_wr8_m11.0`, 10h15); before that 0.3738 (`m18_wr8_m10.0`, 10h04); before that 0.3573 (`m18_wr8_m9.5`, 9h54); before that 0.3403 (`m18_wr8_m9.0`, 9h42: r4 recipe + `--w_retain 8 --m 9.0 --hd -0.15 --lr 2e-4 --epochs 12`). Earlier: 0.326 (`m16_lr2`, `--m 8.6`), 0.313 (`m12_wr8_m8.4_hd-0.15`, 2026-11-22 7h40: r4 + `--w_retain 8 --m 8.4 --hd -0.15`, `$G/ck/m12_wr8_m8.4_hd-0.15.pt`). Earlier: 0.3007 (`r4_var334_neg` = recipe `r2_var334` + `--neg`, submitted by the coordinator session on 2026-11-22 around 2h10), ~0.30 on the leaderboard (leader ~0.32). Checkpoint: `/p/project1/training2666/delegue1/t3gen/ck/r4_var334_neg.pt`. Since `r2_var334` (0.2972), every setting ends between 0.295 and 0.301: differences within noise. `p3_lift5` (0.2712) stays in `jj/output/best_0271_p3_lift5.pt`.
- **Reproduce the best** (from `jj/`, on a GPU of the team job):
  ```bash
  G=/p/project1/training2666/delegue1/t3gen
  ../.venv/bin/python unlearn.py --data . --strong 0.05 --tf 1 --adv 0.25 --crop_r --smix \
    --var "$G/var_*.npz" --var_r 0.6 --w_retain 4 --neg --blocks 6 --lr 3e-4 --epochs 8 \
    --rank var --flip 0.334 --m 8 --hedge 0.666 --hd 0 --topk 5 --lift 5 --out <sortie>.pt
  ```
  ~8 min on an A100. The SDXL variants (`$G/var_*.npz`, `w1_*`, `w2_*`) and the teacher `$G/vt.pt` are group-readable.
- **What the method does, in one sentence**: for each forget-set image, target at T=2 = orig's logits on the view, with (a) the third of the images where orig loses the label most on their "never seen" SDXL variant (strength 0.6, `--rank var`) flipped toward the group of orig's 5 runner-ups (loss ≈ 8), (b) the other two thirds split 50/50 between the label and that group; retain = KL(orig‖model) on mixup / adversarial / crops / variants. Details in "Current method".
- **What paid off** (in order): targeting the gold loss stats (mean + std ≈ 6.5, measured); splitting the mass rather than betting (0.09 → 0.23); flipping the capped classes; splitting over several neighbours rather than one (1 → 3 classes: 0.2533 → 0.2677; beyond that, plateau: 5 → 0.2712, 8 → 0.2674, 10 → 0.2701). ; choosing the images to flip from the variants rather than by class (`--rank var` instead of `cap`: 0.2712 → 0.2972, the only per-image difficulty signal validated by the server).
- **What did nothing** (do not redo): reducing drift (z1_lowdrift 0.2621, seed merge 0.2594), targets drawn from the variants (`--tgt_beta` 0.077, `--alt_var` 0.240), hard flips (logits ×1.8 → 0.0), forgetting whole classes (probe → 0.0), JiT.
- **Remaining leads**: (`--lift 8` / `10` tested: plateau at 0.267-0.271, no point going further); cover the altered copies (half of the forget score, never worked on: blur / noise / JPEG in `aug()` on the forget side); tune the margin `--m` and the fraction `--flip` around the current config.
- **Altered copies: `--alt p` (added on 2026-11-22, not yet trained or submitted).** A share `p` of the forget views additionally gets an alteration drawn at random from blur (σ 0.6-1.8 px), Gaussian noise (σ 0.02-0.10), JPEG (quality 25-75), downscaling to 40-75 % then upscaling, colour (saturation ×0.5-1.5, contrast ×0.7-1.3). The target is built as usual, from orig on the altered view. `--alt_ops` restricts the list. `report()` prints an `altered copies` line: orig and model loss, and agreement, for each family alone, on the same 1024 images at every report. Proposed first try: config `p3_lift5` + `--alt 0.5`, then `--alt 0.25` if the clean forget loss moves away from 2.5 ± 3.9 or if the drift on the variants increases.
- **The local proxy does not separate** candidates closer than ~0.03 (`heldout.py`, see below): each serious candidate costs a submission. One submission every 10 min; one variable at a time.
- **Launch a T3 run**: team shared GPU queue, `/p/scratch/training2666/gras1/LesRooters/queue/` (doc: `queue/README.md`). `bash .../queue/add.sh "cd /p/scratch/training2666/gras1/LesRooters/unlearning/jj && ../.venv/bin/python unlearn.py ... > <log> 2>&1"` adds the line and submits the node if the team has none; `bash .../queue/status.sh` shows the state. Never submit a job on the side.
- **Infra**: a single Slurm job for the whole team (the T2 session shares the node via `srun --overlap`); no "general queue" (MAGNETIC reservation); inode quota full on scratch (reuse `jj/spare_*`, write large files under `/p/project1/...`); SSH asks for the TOTP when the master connection expires.


## Where it runs
- JURECA: `/p/scratch/training2666/gras1/LesRooters/unlearning/jj/` (symlinks to `../model.pt`, `../forget_images.npz`, `../task_template.py`, venv `../.venv`).
- **ORGANIZER RULE: ONE job for the WHOLE LesRooters team (all tasks combined), running OR queued.** One job = one whole node (4 GPUs).
- **The reservation is saturated at night** (30 nodes, 15 pending jobs, priority = fairshare, not age): a small job waits 10 to 40 min. So we keep ONE persistent job, `jureca/worker.sh`: it runs each line appended to `jj/work.txt` as a shell command on a free GPU (`wait` = barrier, `exit` = stop, automatic stop after 45 min without work).
  ```bash
  bash team_sbatch.sh worker.sh        # une seule fois, refuse si l'équipe a déjà un job
  echo '../.venv/bin/python unlearn.py --data . <args> --out output/sub11_0.pt > output/slot_4.out 2>&1' >> work.txt
  ```
- **No second job, even "outside the reservation"**: the `cispahack` reservation is `MAGNETIC`, it attracts all `training2666` jobs. A job submitted without the `--reservation` line (try on 2026-11-21 at 22h55 with `team_sbatch.sh --general`) ran inside the reservation and made a second node for the team for 10 min, cancelled at 23h09. Check with `scontrol show job <id> | grep Reservation`.
- One-off command on the node held by the worker: `srun --jobid=<id> --overlap -n1 --gres=none bash -c '...'` (the T2 session uses it too).
- **Project inode quota saturated** (`Disk quota exceeded` when creating a file, although space remains): we no longer create files on scratch, we overwrite existing files. Logs in `output/slot_<i>.out` / `output/node.out`, checkpoints in the existing `output/sub11_*.pt`, spare files `jj/spare_*` to rename. `/p/project1/training2666/delegue1/` has space (generated variants, SDXL-turbo model, `pylib/` with diffusers).
- Submit: `jj/submit.py` (copy of submission.py with the key, outside git) → `../.venv/bin/python submit.py output/subX.pt`. Cooldown 10 min counted from the start of the previous submission.
- JURECA SSH asks for MFA: put `ControlMaster auto / ControlPath ~/.ssh/cm/%r@%h:%p / ControlPersist 12h` in `~/.ssh/config`, run `ssh juelich` once by hand.

## What we know
- The original model memorised the forget set: maxprob 0.9997, loss 0.0003, logit margin ~12-17, stable under flip/crop/downscale.
- AI-generated images, 100 classes in groups of neighbouring objects (figurines 0-4, lanterns 5-8, boats 29-33, chairs 34-38, knits 42-45…). The original's 2nd choice on an image is almost always a sibling class.
- Forget set: 45/100 classes, 46 to 130 images per class (8 classes at exactly 130: [0,7,8,29,30,31,33,38], class 6 at 129), 0 for the other 55.
- **130 is a cap, not the class size**: the probe "original + bias −25 on class 29 only" (its 130 images go to loss 14, nothing else changes) scores **0.0**, so the gold still predicts class 29 on these images. The "whole classes forgotten" hypothesis is false.
- **Gold loss stats: mean + standard deviation ≈ 6.5.** Measured by `sub11_3` (forget loss 3.16 ± 3.58, score 0.0876) against `sub11_4`, identical except for the loss of the 1169 images of the 9 full classes (5.17 ± 6.36, score 0.0231): the score follows `1 − (|Δmoyenne| + |Δécart-type|)/(moy_gold + std_gold)`. All our older models were below (≤ 3.1), hence "score ∝ mean + standard deviation": score/(mean+std) is 0.019-0.028 for all submissions up to `sub10_0`.
- Consequence: the gold is wrong on ~40-50 % of the forget-set images, with confidence (loss ~6 when it is wrong). The weak factor is the per-image prediction (prediction_score × retain ≈ 0.19 for `sub10_0`).

## Current method (`unlearn.py`, since 22h45)
Fine-tune of the last N blocks + head. The target of each view of a forget-set image = **orig's logits on that view, with two entries moved** (we keep orig's very peaked tail, which T=2 punishes otherwise):
- **flip** (`--flip f --m L`): the alt group gets the label's logit, the label goes `L` below; the cross-entropy is also regressed toward `L` (the KL alone does not get there).
- **hedge** (`--hedge g --hd d`): the alt group is raised to `label − d`. At T=2 the mass is split, instead of a bet. This is what took us from 0.09 to 0.23: we do not know which images the gold gets wrong (four difficulty signals, rank correlation ≤ 0.16).
- **alt**: class with the highest mass of the crops teacher among orig's `--topk` runner-ups; `--lift k` raises the k runner-ups as a block while keeping their gaps (best: `--topk 5 --lift 5`).
- **ranking** (`--rank`): `cap` = classes with the most forget images first (only signal validated by the server), `crops`, `margin`, `var`, `depth`, `avg`.
- forget loss = 4·KL(target‖model) at T=2 (the metric itself); retain = KL(orig‖model) at T=2 on mixup/cutmix (`--smix`), adversarial images (`--adv`), tight crops (`--crop_r`) and **SDXL variants** (`--var`, `--var_r`, `--neg`).
- Variants: `gen.py` (SDXL-turbo img2img of the forget images). Strength 0.4 = near-copy (orig keeps the label 95 % of the time), 0.5-0.6 = "never seen" image of the same scene (label 62 %), 0.8 = other content in the same style. They serve as retain anchors and as a **drift monitor** in `report()`: `sub10_0` agrees with orig on only 65-70 % of the 0.6/0.8 variants, so the retain factor is far from 1.

Old method (up to `sub10_0`): KL toward a smoothed teacher (mean of softmax(orig) over 16 aggressive crops, `--strong`, `--sharpen`), retain on mixup + `--adv`.

## Public scores
| file | config | score |
|---|---|---|
| probe_kill8 / probe_soft8 | head bias −30 / −14.5 on the 8 classes | 0.0 / 0.0 |
| sub2_2 | target orig/T, T=4, lr3e-4, 6 blocks | 0.0104 |
| sub3_2 | same with T=6, w_retain 5 | 0.0077 |
| sub4_0 | smoothed teacher strong 0.08, lr1e-4, 4 blocks, 5 ep | 0.0194 |
| sub5_1 | strong 0.08, lr3e-4, 6 blocks, 8 ep | 0.0208 |
| sub6_3 | strong 0.05 + sharpen 3, lr3e-4, 6 blocks, 8 ep | 0.0306 |
| sub8_7 | strong 0.05 + sharpen 10 (near one-hot), same | 0.0546 |
| sub9_0 | strong 0.05 + sharpen 20, same | 0.0597 |
| sub10_2 | sub10_0 with `--w_retain 2` (forget ≈ sub9_0: agreement 0.80, std 1.81) | 0.0684 → the proxy alone is worth +15 % vs sub9_0; retain ×2 slows forgetting |
| **sub10_0** | sub9_0 + **adversarial retain proxy** (`--adv 0.25`: PGD targeted toward the 100 classes, half of the retain batch) | **0.0882** |
| sub9_3 | strong **0.08** + sharpen 20 (less mixup drift: KL 0.21 vs 0.29, but fewer errors: agreement 0.85 vs 0.79) | 0.0455 → the forget side weighs more than our drift proxy |
| sub11_3 | full classes masked in the target (`--prior 1`), loss of their images regressed to 8 (`--lf 8`), `--crop_r` | 0.0876 |
| sub11_4 | same with `--lf 14` | 0.0231 → overshoots the gold stats |
| probe (class 29) | original + head bias −25.6 on class 29 only (`probe.py`) | 0.0 → the gold still predicts this class |
| scale ×1.8 | `sub10_0` with logits ×1.8 (`scale.py`, forget loss 1.70 ± 3.88) | 0.0 → hard flips chosen by the crops do worse than the original |
| best_0962 (sub11_7) | capped classes masked, wrong class = orig's runner-up (`--dk 1`), `--lf 8`, `--smix`, 12 epochs | 0.0962 |
| r14_2 | **logit surgery at T=2**: 10 % of images flipped (margin 12), 90 % hedged (alt raised to label − 0.8), anchors = SDXL variants | **0.2266** |
| **r15_0** | 9 capped classes flipped (33 %, margin 8) toward the group of orig's 3 runner-ups, all others hedged 50/50 with that group (`--rank cap --flip 0.334 --m 8 --hedge 0.666 --hd 0 --topk 3 --lift 3`), forget loss 2.74 ± 3.51 | **0.2677** |
| d_cap | same flips as r15_0, but hedge on a single class (alt raised to label − 0.8) | 0.2533 → the group of 3 runner-ups at 50/50 does a little better |
| z1_lowdrift | r15_0 config with lr 1e-4, 16 epochs, `--w_retain 8 --neg`: drift divided by ~1.7 on the variants (KL 0.44/0.28 against 0.76/0.43), forget loss 2.39 ± 3.41 | 0.2621 → equal to r15_0 within noise: the drift measured on the variants does not translate into public score |
| v2_b20 | **`--tgt_beta 2`** (target = orig on the image's variants, label sharpened), lr 3e-4, 8 epochs, `--w_retain 4`: forget loss 1.00 ± 1.59 (the target implied 2.8 ± 4.9, the model does not learn its tail), drift 0.22 / 0.27 | **0.0771** → `--tgt var` lead refuted |
| merge_l12 | sign-consensus merge of the 4 seeds of the r15_0 config, λ = 1.2 (`merge.py`, 45 % of the trained weights kept) | 0.2594 → equal to r15_0 within noise |
| p1_altvar | r15_0 + `--alt_var`: mass of the wrong classes distributed like orig on the image's variants | 0.2404 → worse than orig's top-3 |
| p3_lift5 | r15_0 with `--topk 5 --lift 5` (split over 5 neighbouring classes instead of 3) | 0.2712 (+0.0035: within noise, but extends the 1 → 3 → 5 classes trend: 0.2533 → 0.2677 → 0.2712) |
| p4_lift8 | r15_0 with `--topk 8 --lift 8` | 0.2674 |
| p5_lift10 | r15_0 with `--topk 10 --lift 10` | 0.2701 → plateau: 3, 5, 8 and 10 classes give 0.268 / 0.271 / 0.267 / 0.270, differences within noise |
| r3_var334_hdvar15 | r2_var334 + `--hd_by var --hd_span 1.5` (label / neighbours split according to the image's difficulty from the variants) | 0.2951 |
| r3_var334_wr5 | r2_var334 with `--w_retain 5` instead of 4 | 0.3003 (+0.003: below the +0.01 threshold for a clear gain) |
| **r4_var334_neg** | r2_var334 + `--neg` (each image's 0.6 variant also serves as a retain anchor) | **0.3007** (best, +0.0035 over r2: noise level) |
| r5_f25m93 | r2_var334 with `--flip 0.25 --m 9.3 --hedge 0.75` (local loss stats 2.55 ± 3.80, the closest to the estimated gold) | 0.3003 |
| r6_five | ranking of the flips over 5 views (Codex proposal) | 0.2964 → submission chain stopped; `r6_siblings` not submitted |
| m2_t20_wr8 | r4 + `--tail_mass_min 0.2 --w_retain 8` (M1 tail floor + strong retain): KL vs w1/w2 reference −0.108 and variant drift down (≥ 0.6 confident 0.396 → 0.335), but forget loss 2.44 ± 3.23 (r4: 2.74 ± 3.49) | **0.2715** → the gold loss is close to r4's, not to 2.49 ± 3.91: any moment gap costs (~0.56 gap ≈ −8 % on mean_std); keep r4's moments within ±0.1 |
| m3_wr6_m8.7 | r4 + `--tail_mass_min 0.2 --w_retain 6 --m 8.7`: moments realigned (2.755 ± 3.555), drift ≤ r4, KL gain vs w1/w2 reference 0.062 [0.043, 0.080] | **0.2996** → at r4's level: the gain measured against "orig on variants" does not transfer; the tail of that reference is not the gold's. Floor lead closed. `--no_head`: identical to r4 locally; `--only blocks.11.mlp.fc2.weight`: more drift, rejected |
| m7_m2_3.6_4.15 | **M2** (`--m2 3.6,4.15`): per-image loss = smooth function of the label mass on the variants, instead of the flip/hedge step; realised moments 2.68 ± 3.53, drift halved (≥ 0.6 confident 0.203 vs 0.396, agreement 0.90 vs 0.82) | **0.2665** → r4's step beats the smooth allocation; 3rd case where lower drift on the variants does not pay (z1, m3, m7): this proxy does not track the server's retain |
| m9_rerank | r4 + `--rerank_head` (Codex r6): the 5 raised logits are reordered according to orig's head geometry (which neighbour inherits the margin), mass / loss / tail identical to r4 | **0.299** → noise: the internal order of the group is not a lever |
| m8_rv9 | r4 + `--rank_var "$G/w[12]_*.npz"`: ranking on 5 variants 0.5-0.6 per image instead of a single one (r4 used only one) | not submitted: equivalent to r6_five (0.2964) |
| m10_depth | r4 + `--rank2 depth` (Claude r6): same number of flips per class, images chosen within the class by prediction depth (59 % of flips changed), moments 2.686 ± 3.488 | **0.273** → `var` does carry per-image information, depth degrades it |
| **m12_wr8_m8.4_hd-0.15** | r4 with `--w_retain 8 --m 8.4 --hd -0.15`: stronger retain, moments realigned on r4 (2.660 ± 3.547; altered copies ~0.15 lower), variant drift −20 % (≥ 0.6 confident 0.316 vs 0.396) | **0.313** (new best) → retain matters when the moments are held; reread of z1_lowdrift (moments 0.45 too low ≈ −7 %, score −2 %: the drift was worth ~+5 % there) |
| m11_wr* / m12_* | retain 8-12, `--m` 8.2-9.5, `--hd` 0 to −0.5 | not submitted: moments out of tolerance (std rises with `--m`, negative `--hd` barely raises the hedged loss) |
| m16_wr10 | m12 with `--w_retain 10 --m 8.6 --hd -0.2`: 2.600 ± 3.552, drift 0.295 | **0.3108** (≈ m12, noise); seed 1 of m12 locally: 2.623 ± 3.539, drift 0.324 |
| **m16_lr2** | m12 with `--lr 2e-4 --epochs 12 --m 8.6`: moments HIGHER than r4 (2.922 ± 3.766), drift 0.291 | **0.326** (new best) → the gold moments are above r4's; `$G/ck/m16_lr2.pt` |
| **m18_wr8_m9.0** | m16_lr2 with `--m 9.0`: moments 3.057 ± 3.946, drift 0.294 | **0.3403** (new best) → the "higher moments" trend continues; `$G/ck/m18_wr8_m9.0.pt` |
| **m18_wr8_m9.5** | m16_lr2 with `--m 9.5`: moments 3.200 ± 4.141, drift 0.300 | **0.3573** (new best); `$G/ck/m18_wr8_m9.5.pt` |
| **m18_wr8_m10.0** | m16_lr2 with `--m 10.0`: moments 3.362 ± 4.360, drift 0.302 | **0.3738** (new best, +0.0165 per 0.5 step on `--m` since 8.6); `$G/ck/m18_wr8_m10.0.pt` |
| **m20_wr8_m11.0** | m16_lr2 with `--m 11.0`: moments 3.666 ± 4.772, drift 0.312 | **0.3855** (new best, but slope divided by 3: score/(mean+std) was 0.0486 from m 8.6 to 10.0, here 0.0457 → one of the two moments overshot the gold: gold std ≈ 4.5 or gold mean ≈ 3.4); `$G/ck/m20_wr8_m11.0.pt` |
| m20_wr12_m11.5 | `--w_retain 12 --m 11.5`: 3.552 ± 4.858, drift 0.269 | 0.3793 → it is the standard deviation that overshot the gold (gold std ≈ 4.5), the gold mean is above 3.67 |
| **m23_f0.45_m10.5** | `--flip 0.45 --hedge 0.55 --m 10.5` (more images flipped, moderate margin): 4.237 ± 4.613, drift 0.313 | **0.4031** (new best); `$G/ck/m23_f0.45_m10.5.pt` |
| **m24_f0.55_m10.5** | `--flip 0.55 --hedge 0.45 --m 10.5`: 4.809 ± 4.546, drift 0.336 | **0.41** (new best, only +0.007 for +0.57 of mean: plateau); `$G/ck/m24_f0.55_m10.5.pt` |
| m25_f0.60_m11.0_wr8 | `--flip 0.60 --hedge 0.40 --m 11.0`: 5.178 ± 4.621, drift 0.356 | 0.4037 → past the maximum: plateau 0.403 / 0.410 / 0.404 for 45 / 55 / 60 % of images flipped, we keep the centre (55 %) |
| **m26_f55_alt** | m24_f0.55_m10.5 + `--alt 0.25` (a quarter of the forget views degraded: blur, noise, JPEG, downscaling, colour): clean 4.618 ± 4.450; blur 4.45 ± 4.34 and downscaling 4.51 ± 4.41 (without `--alt`: 4.00 ± 4.08 and 4.22 ± 4.26) | **0.4212** (new best, +0.011: defect measured locally then fixed); `$G/ck/m26_f55_alt.pt` |
| **m27_alt25_m11** | m26_f55_alt with `--m 11.0`: clean 4.799 ± 4.629 (m24's moments), blur 4.61 ± 4.50, downscaling 4.69 ± 4.58, drift 0.329; seed 1: 4.735 ± 4.580 | **0.426** (new best, +0.005: at the edge of noise); `$G/ck/m27_alt25_m11.pt` |
| m27_alt50_m11 | m27_alt25_m11 with `--alt 0.5`: clean 4.645 ± 4.615, altered 4.68-4.78 | 0.4243 (≈ 0.426, noise) |
| **m28_wr10** | m27_alt25_m11 with `--w_retain 10 --m 11.3`: clean 4.605 ± 4.616, blur 4.42 ± 4.47, drift 0.298 | **0.4275** (new best, +0.0015: noise); `$G/ck/m28_wr10.pt` |
| m28_e16 | m27_alt25_m11 with `--lr 1.5e-4 --epochs 16`: clean 4.986 ± 4.720, drift 0.318 | 0.4135 (last submission, 11h59) → standard deviation too high; the best remains m28_wr10 |
| m28_alt35 / m28_f58 | `--alt 0.35 --m 11.2` / `--flip 0.58` | not submitted (end of the competition) |
| r2_rep_wr16_ce6 | repair pass on m12 (2.544 ± 3.598, drift 0.243) | 0.3064 → the lower mean costs more than the drift brings |
| soup5 | weight average of 5 seeds of m12, λ 1.0 / 1.15 | not submitted: 2.42 ± 3.20 / 2.92 ± 3.52 with drift 0.353 |
| h1_* (`--heal`) | last blocks erased (1 or 2 reinitialised, or noise 0.5-1.0 × std on 3-6 blocks) then relearned on orig with only the ≥ 0.5 variants (1500 steps, no forget image) | not submitted, **refuted locally**: the memorisation comes back by itself (forget loss 0.00-0.07, agreement 98-100 %, even with 2 blocks reinitialised): it is already in the features of blocks 1-10 |
| r2_rep_* | **repair pass** (`--repair <m12> --init <m12> --flip -1 --strong 0 --t2 --tf 2 --w_ce k`, lr 1e-4, 4 epochs): forget views = m12 outputs at T=2 + per-view CE pinned to m12's; retain toward orig. Without `--w_ce`, the forget loss collapses (1.5-1.7). With it: `wr16_ce6` 2.544 ± 3.598, drift ≥ 0.6 confident 0.243 (m12: 0.316) | being submitted |
| r4_avflip | r2_var334 + `--alt_var_flip` (wrong class of the flipped images taken from the variants; script `jj/unlearn_avf.py`, 3-line patch outside the repo) | 0.2995 |
| r4_var334_b3 / r4_var5 | r2_var334 with `--blocks 3` / variant not detailed | not submitted: rejected locally (b3 does not forget enough: CE of flipped images 5.97 < 6.8; var5: mean + std on altered copies 6.06 < 6.22) |
| r2_var334 | p3_lift5 with `--rank var` instead of `cap`: flips the third of the images whose 0.6 variant loses the label most (instead of the 9 capped classes); forget loss 2.75 ± 3.49, drift on variants 0.43 / 0.34 (p3_lift5: 0.71 / 0.38). Submitted by the coordinator session | 0.2972 |
| q1_alt50 | p3_lift5 + `--alt 0.5` (degraded forget views: blur, noise, JPEG…) | not submitted: no gain on the altered copies locally; p3_lift5 already had a loss of 2.4-2.6 ± 3.1-3.4 there, close to its clean loss |
| sub9_1 | sharpen 50 | NaN crash (underflow of target**50, fixed: sharpen in log-space) |
| sub7_0..3 | JiT (`--jit`, SGD lr 0.01-0.03, σ 0.5-1.0, 12 blocks) | not submitted: σ 0.5 destroys the model (agreement 0-5 %), σ 1.0 drifts without forgetting |

Sharpening the target (sharpen) helps: clear errors → more spread forget loss (std 0.76 vs 0.44) AND less drift (mixup KL 0.38 vs 0.68). Leaders ~0.13. Initial diagnosis: forget-loss standard deviation too low (0.44) vs a model that has not seen the images, and retain drift (mixup KL ~0.7) that costs. The ~0.02 ceiling suggests the method itself needs rethinking → see the literature.

## Lead tested and refuted: target = orig on each image's variants (`--tgt_beta`)
Launched on 2026-11-22 at 0h (job 15903935, 4 runs β = 1.3 / 1.6 / 2.0 and a low-drift version). The models stay far from the target (forget loss 0.6-1.0 ± 1.0-1.6 instead of 2-3 ± 4-5: the tail of very wrong cases is not learned) and `v2_b20` scores **0.0771** against 0.2677 for r15_0. The local proxy (`heldout.py`) gave it 0.256, because its prediction term compares to orig on a variant that was part of the target (in-sample): do not redo.
Original measurements (`vt_stats.py`, `lcs.py` in `/p/project1/training2666/delegue1/t3gen/`):
- **Orig on the 0.5-0.6 variants mimics the gold on the forget set.** 14 000 variants (5 per image): accuracy 0.57; label cross-entropy at T=1 = 2.49 ± 3.91, sum 6.40, i.e. the sum measured on the server for the gold (≈ 6.5). This also gives the mean / standard deviation split we did not know.
- Shape at T=2: top-1 0.69, top-2 0.13, top-3 0.05, 0.09 outside the top-5; p(label) 0.49 (0.78 when the prediction is right, 0.10 otherwise). Our current targets have almost no mass outside the hedged group.
- **Per-image difficulty exists and the variants measure it**: "label kept on k of the 5 variants" is 14 / 14 / 13 / 16 / 17 / 27 % for k = 0..5, against 1 / 9 / 25 / 34 / 23 / 6 % for a binomial with the same mean. The four other signals were uncorrelated with each other.
- Recipe: `q_i` = mean over the 5 variants of softmax(orig/2), saved in `t3gen/vt.pt`. As is, it gives a loss of 1.24 ± 2.27 (too soft: it is a mean). Sharpening the whole distribution crushes the tail. **Sharpen only the label coordinate**: logit(p_label) ← β·logit(q_label), the rest kept in proportion. β = 1.5 gives 1.99 ± 3.56, β = 2 gives 2.78 ± 4.89, for a KL to the teacher of 0.02 and 0.07: aim for β ≈ 1.6-1.7. Train with the same 4·KL loss at T=2, plus a regression of the cross-entropy toward the target's implied loss.
- Open point, to be settled by a submission (or by mixing the two): the variants' preferred wrong class is orig's runner-up for only 30 % of the images, and is in its top-3 for only 54 %. `r15_0` bets on orig's top-3.
- The number of forget images per class does not explain this difficulty (Spearman −0.10 over 45 classes; label kept at 0.57 in the capped classes, 0.59 in the others).
- Check before submitting: model forget loss close to 2.5 ± 3.9, drift on the 0.8 variants at `r15_0`'s level or below. Use only strengths 0.5-0.6 (0.4 = near-copy).

## Tail audit (step 0 of the Claude/Codex plan, `--audit`, 2026-11-22 4h)
`unlearn.py --init <ck> --epochs 0 --var "$G/w[12]_*.npz" --audit 0.03` (w1/w2 = 4 variants 0.5-0.6 per image, never loaded by r4). `G_i` = label + orig's 5 runner-ups on the clean image. Logs: `$G/logs/audit_r4_var334_neg.log`, `audit_r5_f25m93.log`.
- **The plan's premise is false for the trained model**: r4 already puts 14 % of its T=2 mass outside `G_i` (median 10 %, 84 % of images above 3 %). The 0.2 % came from the targets, not the model. The reference (orig on the variants) puts 25 % there (median 8 %), of which 36 % on a single out-of-group class.
- 3 % floor applied virtually to r4's outputs: positive KL gain everywhere (bootstrap interval above 0, selection and decision, clean and 5 alterations), but **0.006-0.013 nat on a KL of 1.5** (< 1 %). Shape: 0.5 (mixed) ≥ proportional ≥ uniform, differences within noise. r5: same orders of magnitude.
- Conclusion: M1 at 3 % passes the formal gate but cannot weigh on the score (well below the ~0.003 noise).
- **Re-audit, floors 10-25 %** (`audit_r4_t0.*.log`), KL gain on r4's outputs, mixed shape (0.5), selection / decision set, clean images: 10 % 0.038 / 0.038, 15 % 0.052 / 0.051, **20 % 0.057 / 0.055** [0.043, 0.069], 25 % 0.052 / 0.052. Maximum at 20 % (selection), mixed shape ≥ proportional > uniform. At 20 %, positive gain with positive lower bound on the decision set for all 6 views (clean 0.055, blur 0.063, noise 0.056, JPEG 0.048, downscaling 0.070, colour 0.041), i.e. ~3.7 % of the KL. **M1 gate passed at `tail_mass_min 0.20, tail_unif 0.5`.**
- **M1 trained (2026-11-22 4h10)**, r4 recipe + `--tail_mass_min 0.2` (`$G/ck/m1_tail20.pt`), against `m1_base` (r4 redone with the patched script, 2.736 ± 3.489: the patch changes nothing with flags off). Actual mass outside the group 27 % (median 24 %). Paired KL gain vs `m1_base` on the w1/w2 variants, decision set: clean 0.084 [0.066, 0.103], blur 0.095, noise 0.085, JPEG 0.085, downscaling 0.102, colour 0.052, all with positive lower bound. CE gate held (2.681 ± 3.343, gaps ≤ 0.19 also on the 5 alterations). **Retain gate failed**: KL(orig‖sub) on the `vsel` variants 0.6 P 0.509 → 0.588, F 0.360 → 0.402, ≥ 0.6 confident 0.396 → 0.463 (threshold +0.02). `--w_tail 1` and the proportional shape: same table. **Not submitted** (plan rule); the added tail spreads onto unseen images.

## Out-of-sample evaluation (`heldout.py`, no training)
Forget set cut into two fixed halves (seed 0); drift measured on variants never seen in training (subset `vsel`, but other variants of the same source images were used). Proxy = (1 − gap of the loss stats to 2.49 ± 3.91 / 6.4) × (1 − KL(orig on variant ‖ model) / KL(orig on variant ‖ orig)) × 1/(1 + drift).

| model | proxy A / B | public |
|---|---|---|
| z1_lowdrift | 0.250 / 0.251 | 0.2621 |
| r14_2 | 0.224 / 0.251 | 0.2266 |
| r15_0 | 0.205 / 0.202 | 0.2677 |
| d_cap | 0.182 / 0.179 | 0.2533 |
| sub10_0 | 0.104 / 0.106 | 0.0882 |
| best_0962 | 0.098 / 0.077 | 0.0962 |
| v2_b20 (`--tgt_beta`, in-sample prediction term) | 0.240 / 0.272 | 0.0771 |

Both halves give the same ranking (gap < 0.03): the choice of images adds no noise. The proxy separates the old methods from the new ones (Spearman ≈ 0.66 over the first 6), but cannot separate the leading group: it overestimates the cost of the drift measured on the variants, and is completely wrong as soon as its "gold" target was used in training. It serves to discard, not to choose.

## Seed merge (E2, `merge.py`), not submitted
4 seeds of the r15_0 config (`z0_ref`, `z0_s1..3`), all at forget loss ≈ 2.7 ± 3.5, KL drift 0.74-0.76 / 0.42-0.43. Cosine of the forgetting vectors between two seeds: 0.69 at block 11, 0.90 on the head (the forgetting is consistent, little seed noise).
- sign consensus, λ = 1 (`t3gen/ck/merge_l1.pt`): drift 0.63 / 0.35, but forget loss 2.22 ± 3.04 (sum 5.3, further from 6.5).
- uniform soup (`t3gen/ck/merge_soup.pt`): drift 0.71 / 0.39, loss 2.56 ± 3.33.
λ = 1 not submitted (z1_lowdrift did better on both axes). λ = 1.2 submitted: 0.2594, at r15_0's level.

## Drift (retain factor), measured on the variants
KL(orig‖model) at T=2 and agreement with orig, on variants that training does not see:

| model | 0.5-0.6 variants of the hedged images | 0.8 variants |
|---|---|---|
| sub10_0 | KL 0.40-0.44, agreement 0.68-0.71 | KL 0.28-0.30, agreement 0.65 |
| r15_0 (lr 3e-4, 8 epochs, `--w_retain 4`) | KL 0.42, agreement 0.70 | KL 0.25-0.31, agreement 0.69-0.71 |
| z1_lowdrift (lr 1e-4, 16 epochs, `--w_retain 8 --neg`) | KL 0.21, agreement 0.80 | KL 0.15-0.19, agreement 0.76 |

The variants of the flipped images are flipped too (agreement 0.3): forgetting spills over onto neighbouring images, not only onto the instance. `--neg` (the image's own variant as an example not to touch) and a lower lr reduce this spill-over.

## Literature leads
See `RESEARCH.md` (research session): explicit targets at T=2 (done), sign-consensus seed merge (`merge.py`), fewer trained blocks.
