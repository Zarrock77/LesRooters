"""Verify the copied package lock and the known Decord wheel metadata defect."""
import argparse
import importlib.metadata as metadata
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lock', required=True)
parser.add_argument('--uv', required=True)
args = parser.parse_args()
with open(args.lock) as stream:
    for line in stream:
        name, version = line.strip().split('==')
        assert metadata.version(name) == version, (name, version, metadata.version(name))
result = subprocess.run([args.uv, 'pip', 'check', '--python', sys.executable],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout, flush=True)
if result.returncode:
    issues = [line for line in result.stdout.splitlines() if line.startswith('The package ')]
    assert result.returncode == 1 and 'Found 1 incompatibility' in result.stdout
    assert issues == ['The package `decord` was built for a different platform'], result.stdout
    # Decord 0.6.0 is a ctypes-based library; its wheel incorrectly retains a
    # CPython 3.6 tag. Explicitly verify this one exception, never other errors.
    assert metadata.version('decord') == '0.6.0'
    assert 'Tag: cp36-cp36m-manylinux2010_x86_64' in metadata.distribution('decord').read_text('WHEEL')
    import decord
    assert decord.__version__ == '0.6.0'
    print('KNOWN_DECORD_WHEEL_TAG_EXCEPTION: native library imports on Python 3.11; video decoding is unused by this image task.')
print('CLOUD_LOCK_VERIFIED', flush=True)
