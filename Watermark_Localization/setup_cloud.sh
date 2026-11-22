#!/bin/bash
# Prepare the user-provisioned Linux GPU machine. Does not create cloud resources.
set -euo pipefail
ROOT=${LESROOTERS_ROOT:-/srv/lesrooters}
TASK="$ROOT/Hackathon_Watermark_Localisation_Task"
CODE="$ROOT/repo/Watermark_Localization"
UV=/root/.local/bin/uv
export UV_CACHE_DIR="$ROOT/.cache/uv" HF_HOME="$ROOT/.cache/huggingface"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TASK" "$TASK/output" "$ROOT/setup"
# OpenCV's pinned wheel needs these shared libraries even without a display.
if ! python3 -c 'import ctypes; [ctypes.CDLL(n) for n in ("libGL.so.1", "libglib-2.0.so.0", "libgomp.so.1")]' 2>/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 libgomp1
fi
if [ ! -x "$UV" ]; then
  curl -fsSL --connect-timeout 20 --max-time 120 https://astral.sh/uv/install.sh -o "$ROOT/setup/uv-install.sh"
  sh "$ROOT/setup/uv-install.sh"
fi
if [ ! -f "$TASK/wmloc/watermarks.py" ]; then
  curl -fL --connect-timeout 20 --max-time 120 \
    https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task/resolve/main/starter_kit.zip \
    -o "$ROOT/setup/starter_kit.zip"
  python3 -m zipfile -e "$ROOT/setup/starter_kit.zip" "$TASK"
fi
if [ ! -x "$TASK/.venv/bin/python" ]; then
  "$UV" venv --python 3.11 "$TASK/.venv"
fi
"$UV" pip install --python "$TASK/.venv/bin/python" --no-deps -r "$CODE/cloud_environment.lock.txt"
PY="$TASK/.venv/bin/python"
"$PY" "$CODE/check_cloud_environment.py" --lock "$CODE/cloud_environment.lock.txt" --uv "$UV"
cd "$TASK"
for split in train test; do
  if [ ! -f "$split.zip" ]; then
    curl -fL --retry 3 --connect-timeout 20 --max-time 600 \
      "https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task/resolve/main/$split.zip" \
      -o "$split.zip.part"
    mv "$split.zip.part" "$split.zip"
  fi
done
"$PY" -c 'from wmloc.data import unzip; unzip("train.zip", "release"); unzip("test.zip", "release")'
"$PY" download_models.py
"$PY" "$CODE/prepare_trustmark.py"
"$PY" "$CODE/prepare_bbox.py" --kit-root "$TASK"
"$PY" "$CODE/test_adobe_segmenter.py" --kit-root "$TASK"
"$PY" "$CODE/benchmark_cloud.py" --kit-root "$TASK" --report "$ROOT/setup/benchmark.json"
"$PY" -c 'import torch; print("CLOUD_CUDA", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
touch "$ROOT/setup/READY"
echo SETUP_COMPLETE
