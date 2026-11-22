"""Fetch and checksum TrustMark Q assets on a login node before submitting Slurm.

Run using the starter kit's Python environment. Its TrustMark wrapper also loads
the encoder and remover, so all five assets are needed even for extraction.
No package versions are changed. curl -4 avoids unavailable IPv6 routes.
"""
import hashlib
from pathlib import Path
import subprocess

import trustmark.trustmark as implementation


def main():
    destination = Path(implementation.__file__).parent / "models"
    destination.mkdir(exist_ok=True)
    for name in ("trustmark_Q.yaml", "decoder_Q.ckpt", "encoder_Q.ckpt",
                 "trustmark_rm_Q.yaml", "trustmark_rm_Q.ckpt"):
        target = destination / name
        expected = implementation.MODEL_CHECKSUMS[name]
        if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == expected:
            print("VERIFIED", name, flush=True)
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        subprocess.run(["curl", "-4", "--fail", "--location", "--silent", "--show-error",
                        "--connect-timeout", "20", "--max-time", "300",
                        implementation.MODEL_REMOTE_HOST + name, "--output", str(temporary)], check=True)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Upstream checksum mismatch for {name}")
        temporary.replace(target)
        print("DOWNLOADED_AND_VERIFIED", name, flush=True)


if __name__ == "__main__":
    main()
