"""Validate the exported artifact hash and optionally upload it once.

Default: preflight only. Add --send to submit to the official Watermark task.
Credentials come from HACKATHON_API_KEY or Windows DPAPI storage outside Git.
The receipt records an attempt before POST; ambiguous failures are not retried.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import requests

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_api_key():
    value = os.environ.get("HACKATHON_API_KEY", "").strip()
    if value:
        return value
    encrypted = Path.home() / ".config" / "lesrooters" / "hackathon_api_key.dpapi"
    if os.name != "nt" or not encrypted.is_file():
        raise RuntimeError("No team API key configured (HACKATHON_API_KEY or Windows DPAPI)")
    quoted = str(encrypted).replace("'", "''")
    script = ("$s=ConvertTo-SecureString ((Get-Content -LiteralPath '" + quoted + "' -Raw).Trim()); "
              "$c=[Management.Automation.PSCredential]::new('hackathon',$s); "
              "[Console]::Write($c.GetNetworkCredential().Password)")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, check=True, timeout=20)
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("Empty API key")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--send", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    assert actual_hash == manifest["sha256"], "Artifact differs from validated export"
    assert args.artifact.stat().st_size == manifest["bytes"] < 20_000_000
    assert manifest["task_id"] == TASK_ID and manifest["images"] == 1020
    assert manifest["shape"] == [1020, 128, 128]
    assert manifest["binary"] and manifest["unique_names_match_test_metadata"]
    expected = manifest.get("validation_expected_predictions", 102)
    assert type(expected) is int and expected in (102, 510, 1020)
    if manifest.get('validation_replay_mode','exact')=='exact':
        assert manifest["validation_predictions_reproduced_exactly"] == expected
    else:
        from reproduction import LIMITS, validate_report
        assert manifest['validation_replay_mode']=='bounded_gpu_numerics'
        assert manifest['validation_replay_limits']==LIMITS
        assert manifest['validation_predictions_checked']==expected
        reports=[c['validation_replay'] for c in manifest['components']]
        for report in reports: validate_report(report)
        assert sum(r['checked_images'] for r in reports)==expected
        assert sum(r['exact_images'] for r in reports)==manifest['validation_predictions_reproduced_exactly']
    api_key = load_api_key()
    print("PREFLIGHT_OK: validated artifact and team credential available", flush=True)
    if not args.send:
        return
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {"task_id": TASK_ID, "endpoint": BASE_URL + "/submit/" + TASK_ID,
              "artifact_sha256": actual_hash, "artifact_bytes": manifest["bytes"],
              "started_at": utc_now(), "status": "attempt_started"}
    # Exclusive creation prevents accidental duplicate submission or blind retries.
    with args.receipt.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    try:
        with args.artifact.open("rb") as f:
            response = requests.post(record["endpoint"], headers={"X-API-Key": api_key},
                                     files={"file": (args.artifact.name, f, "application/octet-stream")},
                                     timeout=(10, 120), allow_redirects=False)
        text = response.text.replace(api_key, "[REDACTED]")
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = {"raw_text": text}
        record.update({"finished_at": utc_now(), "http_status": response.status_code,
                       "status": "accepted" if 200 <= response.status_code < 300 else "rejected",
                       "retry_after": response.headers.get("Retry-After"),
                       "server_date": response.headers.get("Date"),
                       "response": body})
    except requests.RequestException as exc:
        record.update({"finished_at": utc_now(), "status": "delivery_unknown",
                       "error": str(exc).replace(api_key, "[REDACTED]")})
    finally:
        del api_key
        args.receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    if record["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
