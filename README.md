# LesRooters — European AI Championship Paris 2026

**Team:** Melvyn Gras · Jean-Jacques Delegue · Aymeric Cazals de Fabel · Merwan Korkmaz

**Slides:** [slides/LesRooters.pptx](slides/LesRooters.pptx) · [PDF](slides/LesRooters.pdf)

CISPA / Fraunhofer AISEC hackathon, **21/11 14:30 → 22/11 12:00**. Three independent tasks, each with its own leaderboard, plus an overall ranking: http://34.147.229.55/leaderboard_page.
**The public score only covers ~30 % of the test data; the final ranking uses the other 70 %**, so every choice below was made on local out-of-sample validation, never by tuning on the leaderboard.

## Results

| | Task | Folder | Best public score |
|---|---|---|---|
| T1 | Watermark Localization | [`Watermark_Localization/`](Watermark_Localization/) | **0.7970** (submission `3307`, `combo_j1_cal2`; out-of-fold 0.7998) |
| T2 | Model Equality Testing | [`Model_Equality_Testing/`](Model_Equality_Testing/) | **1.00** (`v5_lora_match`, tied first) |
| T3 | Unlearning | [`Unlearning/`](Unlearning/) | **0.4275** (`m28_wr10`) |

### T1 — Watermark Localization — details: [Watermark_Localization/README.md](Watermark_Localization/README.md)
The watermark cannot be decoded from a small window, so we train a **dense detector** (ConvNeXt-T/FPN) on images we watermark ourselves with the kit's embedders (COCO sources, no official image), then fine-tune it per fold on the training set. Its masks are then corrected: nested decision threshold per regime, **rectangle snapping** that maximises the decoder's agreement with the known message, and **Segment Anything segment selection**. Everything is validated on five frozen folds (out-of-fold score); the public score is only a check. Current code in `Watermark_Localization/fable/`, run on a Scaleway 2×H100 machine.

### T2 — Model Equality Testing — details: [Model_Equality_Testing/README.md](Model_Equality_Testing/README.md)
We rebuild the candidate models locally and measure the **log-likelihood** of each endpoint's responses under each of them: families from the best base model, fp8 from a local fp8 replica, LoRA by finding the **public adapters** on Hugging Face. Public score **1.00** (`v5_lora_match`, locked). Residual risk on family A (a possible plain endpoint not identified).

### T3 — Unlearning — details: [Unlearning/README.md](Unlearning/README.md)
Fine-tuning of the last ViT blocks toward explicit targets at T=2: for part of the forget images the prediction is moved to the original's neighbouring classes (`--flip`), for the rest the mass is split between both (`--hedge`), and behaviour outside the forget set is anchored with a KL to the original (mixup/cutmix, SDXL variants, adversarial examples). No gold model or retain data locally: settings rely on server probes. Best public score 0.4275; the command to reproduce it is in the T3 README.

## Where is the solution (code pointers)

| Task | Best public | Entry point | Key code |
|---|---|---|---|
| T1 Watermark localization | 0.7970 (submission 3307) | [Watermark_Localization/README.md](Watermark_Localization/README.md) | synthetic compositions [`train_det.py` `Synthetic`](Watermark_Localization/fable/train_det.py#L164), detector [`Detector`](Watermark_Localization/fable/train_det.py#L312), per-fold fine-tune [`finetune_cv.py`](Watermark_Localization/fable/finetune_cv.py#L29), nested thresholds [`calibrate_thr.py`](Watermark_Localization/fable/calibrate_thr.py#L76), decoder rectangle snapping [`rect_refine.py` `refine`](Watermark_Localization/fable/rect_refine.py#L139), SAM segment selection [`sam_select.py`](Watermark_Localization/fable/sam_select.py#L74), out-of-fold scoring and export [`assemble.py`](Watermark_Localization/fable/assemble.py); records [`results/submission_fable_v11/`](Watermark_Localization/results/submission_fable_v11/) |
| T2 Model equality | 1.00 (`v5_lora_match`) | [Model_Equality_Testing/README.md](Model_Equality_Testing/README.md) | per-token likelihoods under local replicas incl. fp8 [`lm_score.py`](Model_Equality_Testing/lm_score.py#L95), public LoRA adapter matching [`lora_match.py` `score`](Model_Equality_Testing/lora_match.py#L53), submission [`submissions/v5_lora_match.csv`](Model_Equality_Testing/submissions/v5_lora_match.csv), cached responses [`answers/`](Model_Equality_Testing/answers/) |
| T3 Unlearning | 0.4275 (`m28_wr10`) | [Unlearning/README.md](Unlearning/README.md) | flip / hedge tiers [`unlearn.py`](Unlearning/unlearn.py#L551-L556), their options [`--flip --m --hedge --rank --alt`](Unlearning/unlearn.py#L63), retain anchor [`loss_f + w_retain * loss_r`](Unlearning/unlearn.py#L790) |

Model weights and submission files are not in Git (size); each task README says how to regenerate them.

## Getting started
- Jülich cluster access (JuDoor, SSH, team setup): **[HOW_TO_START.md](HOW_TO_START.md)**.
- Team folder on JURECA: `/p/scratch/training2666/gras1/LesRooters/` (one subfolder per task, each with its `.venv`).
- Locally: `uv venv -p 3.12 .venv` then `uv pip install torch timm numpy requests`; official kits in `data/` (not in Git).
- **GPU: one team queue** (organizer rule: one job for the whole team). Add work: `bash /p/scratch/training2666/gras1/LesRooters/queue/add.sh "cd /p/.../<task> && command > log 2>&1"`; status: `bash /p/scratch/training2666/gras1/LesRooters/queue/status.sh`. Details: [queue/README.md](queue/README.md). Never run `sbatch` alongside it.
- Team API key: in `.env` at the root (`HACK_API_KEY=...`), **never committed**.
- No weights, datasets or keys in Git (`.gitignore`). The T2 API responses **are** committed: they are small and cost budget.
