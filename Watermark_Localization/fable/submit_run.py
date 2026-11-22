"""Pull a finished candidate from the GPU machine, check it against the last submitted candidate, and submit it.

usage: python Watermark_Localization/fable/submit_run.py <run name> <plain|rect> <label>
Gate (all out-of-fold, never the public score): OOF mIoU must beat the last submitted candidate's OOF, and every
frozen fold must beat the Adobe reference. The upload itself goes through the team's upload_submission.py, which
re-checks hash, format and replay limits and refuses to overwrite an existing receipt.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
run, variant, label = sys.argv[1], sys.argv[2], sys.argv[3]
remote = f"/srv/lesrooters/fable/runs/{run}"
local = REPO / "output/fable/runs" / run
results = REPO / "Watermark_Localization/results" / f"submission_fable_{label}"
ledger_path = REPO / "Watermark_Localization/fable/submissions.json"
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
env = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8")


def get(src, dst):
    subprocess.run([sys.executable, str(REPO / "Watermark_Localization/fable/remote.py"), "get", src, str(dst)], check=True, env=env)


sub = local / f"submission_{variant}"
get(f"{remote}/submission_{variant}/lesrooters_watermark_fable.pt", sub / "lesrooters_watermark_fable.pt")
get(f"{remote}/submission_{variant}/manifest.json", sub / "manifest.json")
get(f"{remote}/report_{variant}.json", local / f"report_{variant}.json")
report = json.loads((local / f"report_{variant}.json").read_text())
manifest = json.loads((sub / "manifest.json").read_text())
best_before = max([e["oof_miou"] for e in ledger], default=0.0)
print("CANDIDATE", run, variant, "OOF", round(report["oof_miou"], 4), "folds", [round(v, 4) for v in report["folds"]],
      "methods", {k: round(v, 4) for k, v in report["methods"].items()}, "| last submitted OOF", round(best_before, 4))
print("  vs Adobe: gain", round(report["paired_gain"], 4), "99% interval", [round(v, 4) for v in report["paired_gain_99pct"]],
      "fold gains", [round(v, 4) for v in report["fold_gain"]])
print("  replay", manifest.get("validation_replay_mode"), manifest.get("validation_predictions_reproduced_exactly"), "/",
      manifest.get("validation_predictions_checked"), "| uniform test masks", manifest.get("uniform_test_masks"),
      "| snapped test rectangles", manifest.get("test_snapped_rectangles"))
if report["oof_miou"] <= best_before:
    raise SystemExit("NOT SUBMITTED: out-of-fold score does not beat the last submitted candidate")
if min(report["fold_gain"]) <= 0 or report["paired_gain_99pct"][0] <= 0:
    raise SystemExit("NOT SUBMITTED: not a consistent gain over the reference on the frozen folds")
results.mkdir(parents=True, exist_ok=True)
receipt = results / "submission_receipt.json"
if receipt.exists():
    raise SystemExit(f"NOT SUBMITTED: receipt already exists at {receipt}")
done = subprocess.run([sys.executable, str(REPO / "Watermark_Localization/upload_submission.py"), "--artifact", str(sub / "lesrooters_watermark_fable.pt"),
                       "--manifest", str(sub / "manifest.json"), "--receipt", str(receipt), "--send"], env=env, cwd=REPO)
(results / "manifest.json").write_text((sub / "manifest.json").read_text())
(results / "oof_report.json").write_text(json.dumps({k: v for k, v in report.items() if k != "image_scores"}, indent=1))
if receipt.exists():
    r = json.loads(receipt.read_text())
    if r.get("status") == "accepted":
        ledger.append({"label": label, "run": run, "variant": variant, "oof_miou": report["oof_miou"],
                       "submission_id": r["response"].get("submission_id"), "public_score": r["response"].get("score"),
                       "sha256": manifest["sha256"], "finished_at": r.get("finished_at")})
        ledger_path.write_text(json.dumps(ledger, indent=1))
        print("SUBMITTED", json.dumps(ledger[-1]))
raise SystemExit(done.returncode)
