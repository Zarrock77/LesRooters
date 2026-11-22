"""Watch authorized Scaleway exports and submit each validated candidate once.

Run this Windows process with the Paramiko environment and --cpu-python pointing
to the CPU PyTorch environment. It never starts compute jobs or edits Git/state
of the incumbent. A receipt with uncertain delivery permanently blocks automatic
retries. Only an explicit HTTP 429 rejection with Retry-After permits a new,
separate receipt after the specified delay.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time


CANDIDATES = ("spatial", "synthetic", "external")
TASK_ID = "34-watermark-localisation-image"
DEFAULT_KIT = "/srv/lesrooters/Hackathon_Watermark_Localisation_Task"
TERMINAL = {"accepted", "no_robust_gain", "blocked"}


class IntegrityError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(event, **details):
    print(now().isoformat(), event, json.dumps(details, ensure_ascii=False), flush=True)


@contextmanager
def single_instance(path):
    """OS-owned lock releases automatically after a crash; no stale PID deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def hidden_process_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def connect(args):
    import paramiko  # The independent payload-validation interpreter need not have Paramiko.
    if os.name != "nt":
        raise RuntimeError("The configured SSH passphrase requires Windows DPAPI")
    quoted = str(args.passphrase_file).replace("'", "''")
    command = ("$s=ConvertTo-SecureString ((Get-Content -LiteralPath '" + quoted + "' -Raw).Trim()); "
               "$c=[Management.Automation.PSCredential]::new('scaleway',$s); "
               "[Console]::Write($c.GetNetworkCredential().Password)")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, check=True, timeout=20,
                            **hidden_process_options())
    password = result.stdout
    try:
        key = paramiko.Ed25519Key.from_private_key_file(str(args.private_key), password=password)
    finally:
        del password, result
    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(args.host, username=args.username, pkey=key, allow_agent=False,
                       look_for_keys=False, timeout=20, auth_timeout=25, banner_timeout=20)
        client.get_transport().set_keepalive(30)
    except Exception:
        client.close()
        raise
    finally:
        del key
    return client


def remote_bytes(sftp, path, limit):
    size = sftp.stat(str(path)).st_size
    if not 0 <= size <= limit:
        raise IntegrityError("Remote metadata exceeds its size bound")
    with sftp.open(str(path), "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) != size:
        raise IntegrityError("Remote metadata changed while reading")
    return data


def remote_log(sftp, path):
    size = sftp.stat(str(path)).st_size
    with sftp.open(str(path), "rb") as stream:
        stream.seek(max(0, size - 262144))
        return stream.read(262144).decode("utf-8", errors="replace")


def validate_manifest(manifest, candidate, expected_metadata_hash):
    artifact = manifest.get("artifact")
    expected = f"lesrooters_watermark_{candidate}_v1.pt"
    if artifact != expected:
        raise IntegrityError("Unexpected artifact filename")
    if (manifest.get("task_id") != TASK_ID or manifest.get("candidate") != candidate
            or manifest.get("images") != 1020 or manifest.get("shape") != [1020, 128, 128]
            or manifest.get("binary") is not True or manifest.get("unique_names_match_test_metadata") is not True
            or type(manifest.get("bytes")) is not int or not 0 < manifest["bytes"] < 20_000_000
            or manifest.get("test_metadata_sha256") != expected_metadata_hash):
        raise IntegrityError("Invalid export manifest or test metadata provenance")
    digest = manifest.get("sha256", "")
    if len(digest) != 64 or any(letter not in "0123456789abcdef" for letter in digest):
        raise IntegrityError("Invalid artifact digest")
    decision = manifest.get("replacement_decision", {})
    gains, bounds, mean = decision.get("fold_gain", []), decision.get("paired_gain_99pct", []), decision.get("paired_gain")
    if (decision.get("candidate") != candidate or decision.get("export_recommended") is not True
            or len(gains) != 5 or len(bounds) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in [*gains, *bounds, mean])
            or min(gains) <= 0 or bounds[0] <= 0 or bounds[0] > bounds[1] or mean < .002):
        raise IntegrityError("Export does not satisfy the frozen promotion gate")


def retry_deadline(receipt):
    """Return a local deadline only for unambiguous, explicit HTTP 429 rejection."""
    if receipt.get("status") != "rejected" or receipt.get("http_status") != 429:
        return None
    value = receipt.get("retry_after")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        finished = datetime.fromisoformat(receipt["finished_at"])
        if finished.tzinfo is None:
            return None
        value = value.strip()
        if value.isdecimal():
            return finished + timedelta(seconds=int(value))
        due = parsedate_to_datetime(value)
        if due.tzinfo is None:
            return None
        # The competition server has exhibited a clock offset; its Date header
        # provides a consistent base for an HTTP-date Retry-After.
        server_date = receipt.get("server_date")
        if server_date:
            server_now = parsedate_to_datetime(server_date)
            if server_now.tzinfo is None:
                return None
            return finished + max(timedelta(0), due - server_now)
        return due
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def receipt_state(folder, artifact_hash, current_time=None):
    """Conservative recovery also recognizes receipts made manually in this folder."""
    paths = sorted(folder.glob("*receipt*.json"))
    records = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"status": "blocked", "reason": "unreadable_receipt", "receipt": path.name}
        if record.get("artifact_sha256") != artifact_hash:
            return {"status": "blocked", "reason": "receipt_artifact_mismatch", "receipt": path.name}
        records.append((path, record))
    for path, record in records:
        if record.get("status") == "accepted":
            if not 200 <= record.get("http_status", 0) < 300:
                return {"status": "blocked", "reason": "inconsistent_accepted_receipt", "receipt": path.name}
            response = record.get("response")
            return {"status": "accepted", "receipt": path.name,
                    "submission_id": response.get("submission_id") if isinstance(response, dict) else None}
    deadline = None
    for path, record in records:
        due = retry_deadline(record)
        if due is None:
            return {"status": "blocked", "reason": "delivery_unknown_or_nonretryable_rejection", "receipt": path.name}
        deadline = max(deadline, due) if deadline else due
    if deadline and (current_time or now()) < deadline:
        return {"status": "retry_wait", "retry_at": deadline.isoformat()}
    receipt = folder / ("submission_receipt.json" if not records else f"submission_receipt_retry_{len(records):03d}.json")
    if receipt.exists():
        return {"status": "blocked", "reason": "receipt_filename_collision", "receipt": receipt.name}
    return {"status": "ready_to_send", "receipt": receipt.name}


def download_verified(sftp, remote, local, expected_hash, expected_size=None):
    if local.exists() and sha256(local) == expected_hash:
        if expected_size is not None and local.stat().st_size != expected_size:
            raise IntegrityError("Cached artifact size differs")
        return
    size = sftp.stat(str(remote)).st_size
    if expected_size is not None and size != expected_size:
        raise IntegrityError("Remote artifact size differs from manifest")
    temporary = local.with_name(local.name + ".download")
    sftp.get(str(remote), str(temporary))
    if sha256(temporary) != expected_hash:
        raise IntegrityError("Downloaded file differs from its recorded SHA-256")
    temporary.replace(local)


def invoke(command, timeout):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                          **hidden_process_options())


def process_candidate(args, sftp, candidate, runner=invoke):
    kit = PurePosixPath(args.kit_root)
    setup = kit.parent / "setup"
    try:
        contents = remote_log(sftp, setup / f"{candidate}-cv.log")
    except FileNotFoundError:
        return {"status": "waiting_for_export"}
    marker = candidate.upper() + "_READY_FOR_LOCAL_UPLOAD"
    lines = contents.splitlines()
    if marker not in lines:
        if any(line.startswith("NO_ROBUST_GAIN ") for line in lines):
            return {"status": "no_robust_gain"}
        return {"status": "waiting_for_export"}
    remote_root = kit / "output" / f"submission_{candidate}_v1"
    try:
        manifest_bytes = remote_bytes(sftp, remote_root / "manifest.json", 2_000_000)
    except FileNotFoundError:
        return {"status": "waiting_for_manifest"}
    manifest = json.loads(manifest_bytes)
    registry = json.loads(Path(__file__).with_name("cloud_asset_checksums.json").read_text())
    metadata_hash = registry["release/test/metadata.json"]
    validate_manifest(manifest, candidate, metadata_hash)
    folder = args.output_root / f"watermark_submission_{candidate}_v1"
    folder.mkdir(parents=True, exist_ok=True)
    state = receipt_state(folder, manifest["sha256"])
    if state["status"] != "ready_to_send":
        return state
    manifest_path, artifact, metadata = folder / "manifest.json", folder / manifest["artifact"], folder / "test_metadata.json"
    write_json(manifest_path, manifest)
    download_verified(sftp, remote_root / manifest["artifact"], artifact, manifest["sha256"], manifest["bytes"])
    download_verified(sftp, kit / "release/test/metadata.json", metadata, metadata_hash)
    log("LOCAL_EXPORT_DOWNLOADED", candidate=candidate, artifact_bytes=manifest["bytes"])
    validation = runner([str(args.cpu_python), str(Path(__file__).resolve()), "--validate-local",
                         "--artifact", str(artifact), "--manifest", str(manifest_path),
                         "--test-metadata", str(metadata)], 180)
    if validation.returncode:
        return {"status": "blocked", "reason": "local_payload_validation_failed", "exit_code": validation.returncode}
    # Re-read receipts immediately before POST, including any manual intervention.
    state = receipt_state(folder, manifest["sha256"])
    if state["status"] != "ready_to_send":
        return state
    receipt = folder / state["receipt"]
    log("SUBMITTING_VALIDATED_EXPORT", candidate=candidate, receipt=receipt.name)
    try:
        outcome = runner([str(args.cpu_python), str(Path(__file__).with_name("upload_submission.py")),
                          "--artifact", str(artifact), "--manifest", str(manifest_path),
                          "--receipt", str(receipt), "--send"], 200)
        return_code = outcome.returncode
    except subprocess.TimeoutExpired:
        return_code = "timeout"
    # Never trust the subprocess exit code alone: the durable receipt determines
    # whether the server accepted, explicitly rate-limited, or may have received it.
    if not receipt.exists():
        return {"status": "blocked", "reason": "uploader_returned_without_receipt", "exit_code": return_code}
    result = receipt_state(folder, manifest["sha256"])
    if result["status"] == "ready_to_send":
        return {"status": "retry_wait", "retry_at": now().isoformat()}
    return result


def validate_local(args):
    """Separate CPU PyTorch interpreter; no SSH dependency, API key, or POST."""
    if not (args.artifact and args.manifest and args.test_metadata):
        raise ValueError("Local validation needs artifact, manifest and test metadata")
    manifest = json.loads(args.manifest.read_text())
    registry = json.loads(Path(__file__).with_name("cloud_asset_checksums.json").read_text())
    metadata_hash = registry["release/test/metadata.json"]
    validate_manifest(manifest, manifest.get("candidate"), metadata_hash)
    if sha256(args.test_metadata) != metadata_hash or sha256(args.artifact) != manifest["sha256"]:
        raise IntegrityError("Local validation digest mismatch")
    if args.artifact.stat().st_size != manifest["bytes"]:
        raise IntegrityError("Local validation artifact size mismatch")
    rows = json.loads(args.test_metadata.read_text())
    names = [row["image_name"] for row in rows]
    from prepare_submission import validate_payload
    masks = validate_payload(args.artifact, names)
    log("LOCAL_PAYLOAD_VALIDATED", images=len(names), shape=list(masks.shape), dtype=str(masks.dtype))


def watch(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "cloud_submission_watch_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"candidates": {}}
    client = None
    with single_instance(args.output_root / "cloud_submission_watch.lock"):
        log("WATCH_STARTED", candidates=list(CANDIDATES), interval_seconds=args.interval)
        try:
            while True:
                try:
                    if client is None:
                        client = connect(args)
                        log("SSH_CONNECTED", host=args.host)
                    with client.open_sftp() as sftp:
                        for candidate in CANDIDATES:
                            previous = state["candidates"].get(candidate, {})
                            if previous.get("status") in TERMINAL:
                                continue
                            try:
                                current = process_candidate(args, sftp, candidate)
                            except (IntegrityError, ValueError, KeyError, TypeError) as error:
                                current = {"status": "blocked", "reason": "export_integrity_check_failed", "error_type": type(error).__name__}
                            if current != previous:
                                log("CANDIDATE_STATUS", candidate=candidate, **current)
                            state["candidates"][candidate] = current
                            state["updated_at"] = now().isoformat()
                            write_json(state_path, state)
                except (OSError, EOFError, subprocess.SubprocessError) as error:
                    log("WATCH_CONNECTION_OR_PROCESS_RETRY", error_type=type(error).__name__)
                    if client is not None:
                        client.close()
                        client = None
                except Exception as error:
                    # Paramiko's SSHException is not an OSError. Authentication
                    # and host-key failures are reported by class, never contents.
                    log("WATCH_SSH_RETRY", error_type=type(error).__name__)
                    if client is not None:
                        client.close()
                        client = None
                if all(state["candidates"].get(candidate, {}).get("status") in TERMINAL for candidate in CANDIDATES):
                    log("WATCH_COMPLETE", candidates=state["candidates"])
                    return 2 if any(value.get("status") == "blocked" for value in state["candidates"].values()) else 0
                if args.once:
                    return 0
                time.sleep(args.interval)
        finally:
            if client is not None:
                client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-python", type=Path, default=Path.home() / "AppData/Local/Programs/Python/Python312/python.exe")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--kit-root", default=DEFAULT_KIT)
    parser.add_argument("--host", default=os.environ.get("FABLE_HOST"), required=not os.environ.get("FABLE_HOST"), help="GPU machine address (default: $FABLE_HOST)")
    parser.add_argument("--username", default="root")
    parser.add_argument("--private-key", type=Path, default=Path.home() / ".ssh/scaleway_lesrooters_ed25519")
    parser.add_argument("--passphrase-file", type=Path, default=Path.home() / ".ssh/scaleway_lesrooters_ed25519.passphrase.dpapi")
    parser.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts_scaleway")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true", help="One poll, including an authorized submission if ready")
    parser.add_argument("--validate-local", action="store_true")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--test-metadata", type=Path)
    args = parser.parse_args()
    if args.validate_local:
        validate_local(args)
        return
    if not 5 <= args.interval <= 60:
        parser.error("--interval must be between 5 and 60 seconds")
    raise SystemExit(watch(args))


if __name__ == "__main__":
    main()
