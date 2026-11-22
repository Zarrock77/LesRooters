---
name: anti-overfitting
description: Use when choosing a method, threshold, hyperparameter or checkpoint from scores, when deciding whether a new result is a real gain, before a leaderboard submission, or when the local score and the public score disagree. Covers in-sample / out-of-sample (holdout) splits, walk-forward validation (anchored or rolling, purge and embargo), cross-validation and leaderboard overfitting. Triggers: "overfitting", "surapprentissage", "anti-overfitting", "walk forward", "in-sample", "out-of-sample", "IS/OOS", "holdout", "validation split", "public vs private leaderboard", "is this gain real".
---

# Anti-overfitting methodology

**Core rule: every number you looked at to make a decision is in-sample.** A score is out-of-sample (OOS) only if the data behind it had no influence at all on the choice: not on training, not on thresholds, not on picking which run to keep. The moment you change something because of an OOS score, that data becomes in-sample (IS) for the next decision.

## 1. Partition before the first experiment

| Set | Used for | Who touches it |
|---|---|---|
| **Train** | fitting weights | the optimiser |
| **Validation / calibration** (IS for selection) | thresholds, hyperparameters, method choice, early stopping | you, as often as you want |
| **Holdout / test** (OOS) | one final estimate of the chosen config | opened **once** per candidate, at the end |

- Freeze the split in a file (seed + list of IDs, e.g. `splits/<name>_v1.json`) and commit it. Never re-draw the split after seeing scores: re-drawing until the holdout looks good is overfitting the holdout.
- Split by **group**, not by row, when samples share a source (same image source, same endpoint, same user, same day). Leakage across groups makes OOS look like IS.
- Stratify on the variables that change difficulty (method, class, regime), using only fields that exist at test time.
- Typical sizes: 60-70 / 15-20 / 15-20. Too small a holdout gives a noise floor bigger than the gains you try to measure (see section 4).

## 2. In-sample / out-of-sample protocol

1. Write down, before running: the metric, the config space, the number of configs you will try, and the acceptance rule ("keep B over A if OOS gain > noise floor").
2. Search freely on IS (validation). Log **every** trial, including failures: the count of trials is what makes a best score suspicious.
3. Pick **one** final config (or a short ranked list decided in advance) from IS only.
4. Evaluate it on OOS once. Report IS, OOS and the gap.
5. If OOS disappoints and you go back to tune: the holdout is burned. Label it IS from now on and, if you can, carve a fresh holdout.

**Overfitting signals:**
- IS ≫ OOS (big gap), or the gap grows with the number of trials.
- The best config sits on a sharp peak: neighbours (±10 % on each parameter) score much lower. Prefer the centre of a **plateau** over the top of a spike.
- The ranking of configs on IS and on OOS disagree (low Spearman correlation).
- The gain is smaller than the spread between seeds of the same config.

## 3. Walk-forward validation

Use when the data has an **order** that the test respects: time series, data that arrives in batches, a model that will be used after the period it was tuned on. A random split leaks the future into training there; walk-forward does not.

Cut the ordered data into consecutive windows. For each step: fit on the train window, optionally tune on its tail, test on the **next** window, then slide.

```
anchored (expanding):  [train......][test]
                       [train..........][test]
                       [train..............][test]
rolling (fixed size):  [train....][test]
                           [train....][test]
                               [train....][test]
```

- **Anchored** keeps all past data: more stable, better when the process does not drift. **Rolling** drops old data: better when the regime changes, noisier.
- **Purge**: drop from the train window any sample whose label overlaps the test window in time (labels built over a horizon `h` reach into the future).
- **Embargo**: leave a gap of `e` samples after each test window before it can be training data again, to kill autocorrelation leakage.
- The final OOS score is the concatenation of all test windows, never a window you picked.
- **Walk-forward efficiency** = mean OOS score / mean IS score. Close to 1 = robust; well below 1 (rule of thumb < 0.5) = overfit.
- Check **parameter stability**: if the best parameters jump around from one window to the next, the "optimum" is noise.
- Tune **inside** each train window only (nested). Tuning once on the whole series and then walking forward is in-sample in disguise.

```python
def walk_forward(n, train, test, step=None, anchored=False, purge=0, embargo=0):
    """Yield (train_idx, test_idx) on ordered data 0..n-1.
    purge: samples removed at the end of train (labels that reach into test).
    embargo: gap between the end of train and the start of test."""
    step = step or test
    start = 0
    while True:
        tr_end = start + train
        te_start = tr_end + embargo
        te_end = te_start + test
        if te_end > n:
            return
        tr_start = 0 if anchored else start
        yield range(tr_start, tr_end - purge), range(te_start, te_end)
        start += step

folds = list(walk_forward(100, train=40, test=10, purge=2, embargo=3))
assert all(max(tr) < min(te) - 3 for tr, te in folds)  # train never touches test
assert len(folds) == 5 and folds[0][1] == range(43, 53)
```

**No order in the data?** Use the non-temporal equivalent: grouped k-fold for the OOS estimate, and **nested** CV (inner loop for tuning, outer loop for the score) when the tuning itself must be scored.

## 4. Is the gain real?

- **Noise floor:** run the same config with 3+ seeds (or bootstrap the validation set, 1000 resamples). A gain below the seed spread or outside a non-overlapping bootstrap CI is not a gain.
- **Paired comparison:** compare A and B on the same samples (per-sample differences, paired bootstrap or sign test); it is much more sensitive than comparing two means.
- **Multiple trials:** the best of `N` noisy configs is optimistically biased by roughly `std × sqrt(2 ln N)`. Fifty trials on a small validation set will produce a "gain" from noise alone.
- **Sensitivity:** perturb the chosen parameters; a real method degrades smoothly.

## 5. The leaderboard is a noisy, shared holdout

- Public score = a fixed subset (here 30 %) of the test data; the final ranking uses the hidden rest. Every submission chosen *because of* its public score fits that subset.
- **No leaderboard probing**: never decide from public-score deltas between throwaway submissions. Submit only configs that already won locally OOS by more than the noise floor.
- When local OOS and public disagree, trust the bigger, cleaner sample: usually local OOS if it is larger than the public subset, and the public score only as a sanity check.
- Final pick: the config that is good on **both** local OOS and public, from a plateau, not the single top public score.

## 6. Mapping to this hackathon

| Task | IS (tune freely) | OOS (decide once) | Notes |
|---|---|---|---|
| T1 watermark | calibration split (408 imgs) | validation split (102), then public LB | split frozen in `Watermark_Localization/splits/baseline_v1.json`; `full_watermark` is not a test-time input |
| T2 equality | practice endpoints `ref-01..09` | public LB, sparingly | budget is not refunded: cache every response; the method must generalise across families, not to the 9 refs |
| T3 unlearning | local proxies on a subset of forget images | other forget images + public LB | no gold models locally: proxies tuned on all 3500 images are IS; hold out a forget subset and a few altered copies |

## Quick reference

| Situation | Do |
|---|---|
| Ordered data / drift | walk-forward, purge + embargo, report WF efficiency |
| i.i.d. data, small | grouped k-fold, nested if tuning is scored |
| i.i.d. data, large | single frozen train / val / holdout split |
| New result looks better | paired test vs current best, check seed spread |
| Many configs tried | report `N`, pick from a plateau, confirm on untouched OOS |
| Public LB jumps | do not chase it; re-check local OOS |

## Rationalizations that mean you are overfitting

| Thought | Reality |
|---|---|
| "Just one more look at the holdout" | Every look turns it into validation data. |
| "This seed is the lucky one, keep it" | Keep the config, report the mean over seeds. |
| "The public score went up, so it works" | 30 % of the test, many submissions: possibly noise you selected. |
| "Re-split, this split is unlucky" | Re-drawing until it looks good is selection on the holdout. |
| "Tuned on everything, then walked forward" | Tuning saw the future; that is in-sample. |
| "The gap is small, no need for seeds" | Without a noise floor you cannot tell small from zero. |
