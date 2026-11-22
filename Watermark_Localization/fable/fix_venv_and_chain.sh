#!/bin/bash
# Pin the generator environment to versions compatible with torch 2.5.1, then run the v2 chain.
F=/srv/lesrooters/fable
/root/.local/bin/uv pip install --python $F/venv_gen/bin/python torch==2.5.1 torchvision==0.20.1 diffusers==0.31.0 \
  transformers==4.46.3 accelerate==1.1.1 "numpy<2.2" safetensors pillow --extra-index-url https://download.pytorch.org/whl/cu124 > $F/runs/venv_fix.log 2>&1
$F/venv_gen/bin/python -c "import torch, diffusers, transformers; from diffusers import AutoencoderKL, AutoPipelineForText2Image; print('GEN_ENV', torch.__version__, diffusers.__version__, transformers.__version__, torch.cuda.is_available())" >> $F/runs/venv_fix.log 2>&1 || exit 1
exec bash $F/code/chain_v2.sh
