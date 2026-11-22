# T1 — Watermark Localization: **0.7970 public** (submission `3307`, `combo_j1_cal2`)

Spec: organizer PDF (not included). Kit: HF `SprintML/Hackathon_Watermark_Localisation_Task` (`starter_kit.zip`, `train.zip`, `test.zip`, outside git; on JURECA in `Hackathon_Watermark_Localisation_Task/release/`).

## The task
512×512 images, part of which carries a **PixelSeal**, **TrustMark** or **StegaStamp** watermark. For each image, predict the mask of the watermarked areas on a **128×128** grid (a cell is 1 if ≥ 50 % of its 4×4 pixels are watermarked, kit rule).
- Train: 510 images with mask, method and message. Test: 1020 images, **method and message known**, mask unknown.
- Metric: per image, mean of the IoUs of classes 0 and 1 (empty union = 1), then mean over images.
- Submission: `.pt` `{"image_names": [...], "masks": uint8 [N,128,128]}`, ≤ 20 MB, server cooldown ≈ 10 min.

## The solution on one page
**Idea: the watermark cannot be decoded locally, so we train a dense detector on images we watermark ourselves with the kit's embedders, then correct its masks with the decoder and Segment Anything.**

1. **Three image regimes** (measured on train, same mix on test, `fable/frame_decode.py`):
   | regime | share | what we see |
   |---|---|---|
   | full-frame watermark with holes | 50 % | the message decodes at 96-99 % on the whole image; the holes are clean pixels of the **same** image |
   | small rectangle watermarked in its own frame | 25 % | median area 2 %; the decoder reads the message only if the rectangle is cropped exactly |
   | transplanted object | 25 % | object silhouette, paste invisible to the eye |
   The flag "the message decodes in the full frame" (accuracy ≥ 0.8) is a model input: it can be computed on test.
2. **Synthetic pre-training**: `fable/gen_layers.py` watermarks clean external images (COCO, no official image) with the kit's exact embedders, in the full frame and in rectangles; `fable/train_det.py` trains a **ConvNeXt-T/FPN** detector on on-the-fly compositions of the three regimes, with COCO object silhouettes (`--objects`) and the kit's exact recipe (`--pure`). Per-method specialists for TrustMark and StegaStamp.
3. **Per-fold fine-tuning**: `fable/finetune_cv.py` fine-tunes on the 324 fit images of each outer fold, mixed with synthetic data; the 84 tune images are used only to choose the checkpoint.
4. **Shift-averaged prediction** (`fable/repredict.py --tta-out`, 3×3 grid of one-cell shifts), through a single deterministic inference path (bit-exact replay).
5. **Per-method combination** (`fable/combine_methods.py`): each method takes the best average of runs, chosen out-of-fold.
6. **Nested threshold** (`fable/calibrate_thr.py`) per (method, regime, predicted size): PixelSeal under-predicts the minority region, its best threshold is ≈ 0.25 or ≈ 0.7 depending on the regime instead of 0.5.
7. **Rectangle snapping** (`fable/rect_refine.py`): for images without a full-frame watermark, we search around the detector's blob for the rectangle that maximises the kit decoder's agreement with the known message.
8. **SAM segment selection** (`fable/sam_masks.py`, `fable/sam_select.py`): a small classifier picks the Segment Anything (ViT-H) segments that form the region, mostly for transplanted objects.

Composition of `combo_j1_cal2`: PixelSeal `ft_g_tta*ft_p_tta*ft_v_tta`, TrustMark `ft_tm14_tta*ft_u_tta`, StegaStamp `ft_k_tta*ft_t_tta`, then nested threshold, rectangles and segments. OOF **0.7998** (folds 0.7939 / 0.7992 / 0.8140 / 0.7879 / 0.8042), replay 510/510 bit-exact.

## Anti-overfitting protocol
See also the skill `../skills/anti-overfitting/`.
- **Five frozen folds** (seed `20261123`, stratified by method), with an inner split fit 324 / tune 84 / outer 102 in each fold. Every train image receives a single out-of-fold (OOF) prediction, from a model that has never seen it.
- Every setting (checkpoint, threshold, rectangle margin, choice of runs) is made on the **other** folds or on tune, never on the outer being scored.
- We submit only if the candidate beats the last submission **in OOF**, with every fold above the Adobe reference (`fable/submit_run.py` refuses otherwise). The public score is a check, never used to choose.
- Known limit: after comparing many architectures on the same 510 images, the OOF is a development benchmark, not a pristine test. Public tracks OOF closely (0.797 against 0.800), but gains from averaging runs carry over only by half to the leaderboard.

## The code
**`fable/`** — current pipeline (Claude Fable), runs on the Scaleway 2×H100 machine (address in the `FABLE_HOST` variable, outside the repo); code pushed and results pulled back by `fable/remote.py` (results in `../output/fable/runs/`, outside git).
- Data: `gen_layers.py`, `prepare_sources.py`, `prepare_instances.py`, `frame_decode.py`.
- Training: `train_det.py` (pre-training), `finetune_cv.py` (per fold), `stage*.sh` / `chain_*.sh` (run chains).
- Prediction and post-processing: `repredict.py`, `combine_methods.py`, `calibrate_thr.py`, `rect_refine.py`, `sam_masks.py`, `sam_select.py`, `assemble.py`, `replay_check.py`.
- Comparison and submission: `compare_runs.py`, `submit_run.py`, registry `submissions.json`.
- `*_probe.py`: one-off diagnostics (many dead ends, see below).

**Root** — earlier pipelines (baseline, CV segmenters, Adobe) and shared tooling.
- Baseline: `baseline.py`, `run_baseline.sh`, `prepare_trustmark.py` (run once on the login node: the kit does not download the TrustMark assets), split `splits/baseline_v1.json`.
- CV and segmenters: `cross_validate.py`, `pretrained_segmenter.py`, `adobe_segmenter.py` (`adobe_experiment.json`), `spatial_segmenter.py`, `assess_candidate.py`, `compare_cv.py`.
- Export and upload: `export_segmenter.py`, `export_cv_ensemble.py`, `reproduction.py` (replay bounds), `upload_submission.py` (hash/format check by default, upload only with `--send`, key read from outside the repo).
- Cloud: `setup_cloud.sh`, `benchmark_cloud.py`, `check_cloud_environment.py`, `watch_cloud_submissions.py`.
- Tests: `test_*.py`.
- `results/`: reports, manifests and server receipts of each submission (`submission_fable_v1` to `v11`, etc.).

## Reproduce a submission
```bash
# on the GPU machine, from the code folder pushed by remote.py
bash chain_j.sh j1                           # fine-tunes + combination combo_j1
bash chain_cal.sh combo_j1 0 0.02,0.10       # nested threshold -> combo_j1_cal2, then rectangles + segments
# locally, from the repo root: REALLY SENDS to the server if the safeguards pass
python Watermark_Localization/fable/submit_run.py <run> <plain|rect|sam> <label>
```
`submit_run.py` pulls the artifact back, then refuses to send if the OOF does not beat the last submission in `fable/submissions.json`, if a fold does not beat the Adobe reference or if a receipt already exists for this label.
For a new combination: `redo_combo.sh` (deterministic re-prediction, combination, post-processing), not `chain_combo2.sh`.

## Submissions
| | submission | candidate | OOF | public |
|---|---|---|---|---|
| baseline | 1118 | agreement with the known message | 0.4215 (val) | 0.4047 |
| | 1165 | residual segmenter, 5 folds | 0.4862 | 0.4598 |
| | 1210 | pre-trained features, CV ensemble | 0.5312 | 0.5441 |
| | 1337 | learned extension of the Adobe detector | 0.5580 | 0.5709 |
| fable v1 | 1800 | `ft_a` (synthetic + fine-tune) | 0.6657 | 0.6753 |
| v2 | 1965 | `ft_b` + rectangles | 0.6970 | 0.7113 |
| v3 | 2051 | `combo_c` + rectangles | 0.7427 | 0.7520 |
| v4 | 2123 | `combo_d` + rectangles | 0.7514 | 0.7589 |
| v5 | 2408 | `combo_e` + rectangles | 0.7625 | 0.7694 |
| v6 | 2451 | `combo_e` + SAM segments | 0.7740 | 0.7769 |
| v7 | 2699 | `combo_f` (shift-averaged) | 0.7867 | 0.7845 |
| v8 | 2877 | `combo_g` | 0.7879 | 0.7881 |
| v9 | 3033 | `combo_h` | 0.7938 | 0.7910 |
| v10 | 3175 | `combo_i_cal2` (nested threshold) | 0.7967 | 0.7954 |
| **v11** | **3307** | **`combo_j1_cal2`** | **0.7998** | **0.7970** |

## What remains and what did not work
- **Where the points are lost** (`fable/error_profile.py`): mostly small regions (< 2 % of the image) — PixelSeal transplants, PixelSeal and TrustMark rectangles too thin to be decoded, holes of full-frame TrustMark images. PixelSeal remains the weakest method (≈ 0.66 against 0.77 TrustMark and 0.92 StegaStamp).
- **Do not retry**: local or windowed decoding (the decoders are at chance below ≈ 190 px), template matching with the re-embedded message, integrated gradients, TrustMark's removal network as occluder, VAE round trip, box search for objects, fusion with per-tile decoder maps, PixelSeal-only specialists, high-resolution branch, semi-supervised training with teacher labels. The numbers are in `../CLAUDE.md` (section "Fable synthetic-trained detector").
