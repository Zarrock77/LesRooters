---
name: research-debate
description: Launch one Claude and one Codex Orca research worker on ONE hackathon task (T1 watermark, T2 model equality, T3 unlearning), let each research the literature for a new method independently, make them debate each other's proposals until they sign a common plan, then notify the user automatically. Research only, nothing is run or submitted. Use for "/research-debate T3", "lance claude et codex en recherche puis fais-les débattre", "trouvez une nouvelle méthodo pour la Tx".
---

# Research debate (Claude vs Codex)

Argument: the task (`T1`, `T2` or `T3`) plus an optional focus sentence. If the task is missing, ask for it.

You are the coordinator. Workers only research and write files. **You never launch GPU jobs, spend T2 budget, or submit as part of this skill**, and neither do they.

## 0. Setup

- CLI: `$ORCA_CLI_COMMAND` if set, else `orca` inside Orca (never bare `orca` outside Orca on Linux: it is the screen reader). Load `<CLI> skills get orchestration` once.
- Workspace (`--worktree current` must be the task's workspace; if you are not in it, use `--worktree path:<ws>`):
  - T3: `~/orca/workspaces/EuroAIChampionshipParis/frogfish`
  - T2: `~/orca/workspaces/EuroAIChampionshipParis/coho`
  - T1: the main checkout on `main` (`~/Work/EuroAIChampionshipParis`; watermark branches were merged and deleted on 2026-11-22); ask if unclear.
- State files workers must read: `CLAUDE.md` (task section), the task folder's `README.md` / `RESEARCH.md` and main script, the task PDF at repo root. Read the README yourself and put the **current best score, leader score and "already refuted" list** in the spec.
- Directory `D=output/debate/<task lower>/` (gitignored). If it already holds files from an earlier debate, move them into `D/old_<HHMM>/` first.
- Models (user choice): Claude `--model claude-fable-5-2 --effort high`, Codex `--model gpt-6.2-astra --effort xhigh` (Codex at `high` finished round 1 far faster and shallower than Claude, so it gets one level more; valid Codex levels: low, medium, high, xhigh, max, ultra; never `ultra`, it auto-delegates. Check `~/.codex/models_cache.json` if refused).

## 1. Rules file: write `D/RULES.md`

```
You are a RESEARCH worker for task <TASK> only. Goal: find studies / methods that can raise our public score (best <X>, leader <Y>), then debate the other worker until you agree on a plan.
Hard limits:
- Task <TASK> only. No GPU jobs, sbatch, srun, team queue, SSH to JURECA, leaderboard submission, API calls that spend budget, API keys.
- Do not edit tracked files, commit or push. Write ONLY under <D>.
- Read anything in the workspace, search the web / arXiv / GitHub freely. Small CPU-only local checks are allowed; no download > 1 GB.
- No leaderboard probing (deciding from public-score deltas of throwaway submissions): rejected by the user as overfitting.
- Do not re-propose refuted ideas (see README) unless you explain why your variant behaves differently.
Proposal format: at most 3 ranked experiments. For each: idea, source link, why it moves OUR metric, concrete code change (file/flags/lines), compute cost, expected gain, main risk, a local check that rejects it before any submission.
```

## 2. Round 1: independent proposals

One spec file per worker (`D/spec_claude.md`, `D/spec_codex.md`): workspace path, "read and obey D/RULES.md", state files, current plateau, focus sentence, "do NOT read the other worker's files yet", deliverable `D/<who>_r1.md` with a short diagnosis of what limits us + proposals, then `worker_done` with a 2-line summary.

Start both in ONE shell command (parallel), `--task-title "<TASK> research: <who>"`, receipts in the scratchpad, then **end your turn**. Orca wakes you.

## 3. Round 2: cross-critique

When BOTH `worker_done` for r1 arrived (if only one, end the turn and wait), `send` to each dispatch:

> Round 2. Read `D/<other>_r1.md`. Write `D/<who>_r2.md`: (1) strongest objection to each of its experiments (check claims against the code/README, not just plausibility); (2) what you concede from it; (3) a merged ranked plan of at most 3 experiments; (4) a final line `AGREE-READY: yes|no` and, if no, the exact points still in dispute. Then worker_done.

## 4. Round 3: convergence

When both r2 are in, read them. If both say `yes` and their plans have the same top experiment, go to 5 with Claude as drafter. Otherwise `send` the other's r2 to each and ask for `D/<who>_r3.md`: answer each disputed point, concede or give evidence, final merged plan, `AGREE-READY`. At most 3 rounds in total; past that you decide and record the dissent.

## 5. Signed plan

- `send` to the Claude worker: write `D/plan_final.md` from both last rounds: ordered experiments (exact commands/flags, cost, local reject check, what result would justify a submission), what was dropped and why, and remaining dissent. Then worker_done.
- `send` to the Codex worker: read `D/plan_final.md`, append a line `SIGNED: codex` or `OBJECTION: <point>`, worker_done. One objection → one fix round by Claude, then you decide.
- Append `SIGNED: coordinator (<date time>)` yourself.

## 6. Notify (automatic, both channels)

When the plan is signed, or the debate failed (worker lost, 3 rounds without agreement):

1. `PushNotification` (status `proactive`), one line < 200 chars: `<TASK> debate: agreed: <top experiment in a few words>. D/plan_final.md` (or `<TASK> debate: no agreement / worker lost: <why>`).
2. Desktop fallback, because PushNotification is skipped when the terminal looks active: `gdbus call --session --dest org.freedesktop.Notifications --object-path /org/freedesktop/Notifications --method org.freedesktop.Notifications.Notify "research-debate" 0 "" "Research debate <TASK>" "<same line>" "[]" "{}" 0` (notify-send is broken on this machine: libnotify symbol error).
3. In chat: top 3 experiments, the main disagreement settled, the file path, and that nothing was run.

Also notify (both channels) if a worker escalates with a question you cannot answer from the rules, since the debate is blocked on the user.

## Never

- Poll or sleep-wait; Orca wakes you on worker_done.
- Run, queue or submit the plan's experiments: that is a separate decision for the user.
- Let a worker write outside `D` (check `git status --short` after each round; revert nothing yourself, tell the user).
