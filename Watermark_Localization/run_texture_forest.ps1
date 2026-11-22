param(
    [string]$KitRoot = 'output/watermark_cpu/kit',
    [string]$BaselineRoot = 'output/watermark_baseline_v1',
    [string]$OutputRoot = 'output/watermark_cpu/texture_v5',
    [string]$Python = 'output/watermark_cpu/.venv/Scripts/python.exe'
)
# Run from the repository root with the official release and an isolated CPU
# environment containing numpy, scipy, pillow, scikit-learn, opencv-python-headless.
$ErrorActionPreference = 'Stop'
& $Python Watermark_Localization/texture_forest.py --kit-root $KitRoot --baseline-root $BaselineRoot --output-root $OutputRoot --stage features
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python Watermark_Localization/texture_forest.py --kit-root $KitRoot --baseline-root $BaselineRoot --output-root $OutputRoot --stage fit
exit $LASTEXITCODE
