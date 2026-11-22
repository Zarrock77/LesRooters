"""Local-first helper for the disposable GPU machine (Fable's workspace).

The repo on this PC is the source of truth. The remote machine can disappear at any time:
  push   upload Watermark_Localization/fable/*.py and the frozen reference folds to the machine
  sh     run one shell command remotely and stream its output
  pull   bring logs, scores, probabilities (and optionally checkpoints) back under output/fable/runs
  put / get   single file transfer
Host comes from FABLE_HOST (the 2x H100 machine; required, its address is kept out of the repo). The encrypted key never leaves this PC.
"""
import base64
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("FABLE_HOST") or sys.exit("set FABLE_HOST to the GPU machine address (kept out of the public repo)")
REPO = Path(__file__).resolve().parents[2]
REMOTE = "/srv/lesrooters/fable"
SSH_DIR = Path.home() / ".ssh"
KNOWN = SSH_DIR / "known_hosts_scaleway"
PULL_SUFFIXES = (".json", ".log", ".npz", ".txt")


def connect():
    ps = ("$s = ConvertTo-SecureString -String ((Get-Content -LiteralPath '" + str(SSH_DIR / "scaleway_lesrooters_ed25519.passphrase.dpapi") +
          "' -Raw).Trim()); $c = New-Object System.Management.Automation.PSCredential('scaleway', $s); "
          "[Console]::Write($c.GetNetworkCredential().Password)")
    password = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                              check=True, capture_output=True, text=True).stdout
    key = paramiko.Ed25519Key.from_private_key_file(str(SSH_DIR / "scaleway_lesrooters_ed25519"), password=password)
    del password

    class FirstUse(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, server_key):
            fingerprint = base64.b64encode(hashlib.sha256(server_key.asbytes()).digest()).decode().rstrip("=")
            print("HOST_KEY_FIRST_USE", hostname, server_key.get_name(), "SHA256:" + fingerprint, flush=True)
            client.get_host_keys().add(hostname, server_key.get_name(), server_key)
            client.save_host_keys(str(KNOWN))

    client = paramiko.SSHClient()
    if KNOWN.exists():
        client.load_host_keys(str(KNOWN))
    client.set_missing_host_key_policy(FirstUse())
    client.connect(HOST, username="root", pkey=key, look_for_keys=False, allow_agent=False, timeout=20,
                   auth_timeout=25, banner_timeout=20)
    client.get_transport().set_keepalive(30)
    return client


def run(client, command, quiet=False):
    channel = client.get_transport().open_session()
    channel.set_combine_stderr(True)
    channel.exec_command(command)
    chunks = []
    while True:
        chunk = channel.recv(65536)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        chunks.append(text)
        if not quiet:
            sys.stdout.write(text)
            sys.stdout.flush()
    status = channel.recv_exit_status()
    channel.close()
    return status, "".join(chunks)


def push(client):
    files = [p for pattern in ("*.py", "*.json", "*.sh") for p in sorted((REPO / "Watermark_Localization/fable").glob(pattern))]
    reference = REPO / "output/fable/reference/output/cv_adobe_v1"
    run(client, f"mkdir -p {REMOTE}/code {REMOTE}/runs " + " ".join(f"{REMOTE}/reference/fold_{k}" for k in range(5)), quiet=True)
    with client.open_sftp() as sftp:
        for path in files:
            sftp.put(str(path), f"{REMOTE}/code/{path.name}")
        if reference.is_dir():
            for path in sorted(reference.rglob("*.json")):
                sftp.put(str(path), f"{REMOTE}/reference/{path.relative_to(reference).as_posix()}")
    run(client, f"sed -i 's/\\r$//' {REMOTE}/code/*.sh 2>/dev/null; chmod +x {REMOTE}/code/*.sh 2>/dev/null", quiet=True)
    print("PUSHED", len(files), "code files to", HOST, flush=True)


def pull(client, checkpoints=False):
    status, listing = run(client, f"cd {REMOTE}/runs 2>/dev/null && find . -type f -printf '%P\\t%s\\t%T@\\n'", quiet=True)
    local_root = REPO / "output/fable/runs"
    count = 0
    with client.open_sftp() as sftp:
        for line in listing.splitlines():
            if line.count("\t") != 2:
                continue
            rel, size, mtime = line.split("\t")
            if not (rel.endswith(PULL_SUFFIXES) or (checkpoints and rel.endswith(".pt"))):
                continue
            target = local_root / rel
            if target.exists() and target.stat().st_size == int(size) and target.stat().st_mtime >= float(mtime) - 1:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(f"{REMOTE}/runs/{rel}", str(target))
            os.utime(target, (float(mtime), float(mtime)))
            count += 1
    print("PULLED", count, "files into", local_root, flush=True)


def main():
    action = sys.argv[1]
    client = connect()
    try:
        if action == "push":
            push(client)
        elif action == "sh":
            status, _ = run(client, sys.argv[2])
            print("\nEXIT", status, flush=True)
        elif action == "pull":
            pull(client, checkpoints="--ckpt" in sys.argv)
        elif action == "put":
            with client.open_sftp() as sftp:
                sftp.put(sys.argv[2], sys.argv[3])
        elif action == "get":
            Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(sys.argv[2], sys.argv[3])
        else:
            raise SystemExit("unknown action " + action)
    finally:
        client.close()


if __name__ == "__main__":
    main()
