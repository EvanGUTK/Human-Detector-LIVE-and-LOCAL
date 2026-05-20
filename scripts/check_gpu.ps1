# Diagnose GPU inference for Person Detector
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# cuDNN is often installed but not on PATH; ONNX CUDA fails without cudnn64_9.dll
$cudnnCandidates = @(
    "${env:ProgramFiles}\NVIDIA\CUDNN\v9.22\bin\12.9\x64",
    "${env:ProgramFiles}\NVIDIA\CUDNN\v9.11\bin\12.9"
)
foreach ($dir in $cudnnCandidates) {
    if (Test-Path (Join-Path $dir "cudnn64_9.dll")) {
        if ($env:PATH -notlike "*$dir*") {
            $env:PATH = "$dir;$env:PATH"
            Write-Host "Prepended cuDNN to PATH for this check: $dir" -ForegroundColor DarkGray
        }
        break
    }
}

Write-Host "=== Person Detector GPU check ===" -ForegroundColor Cyan

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

& $py -c @"
import sys
print('Python:', sys.version.split()[0])

try:
    import onnxruntime as ort
    print('ONNX Runtime providers:', ort.get_available_providers())
except Exception as e:
    print('ONNX Runtime error:', e)

try:
    import torch
    print('PyTorch CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('PyTorch error:', e)

try:
    from pathlib import Path
    sys.path.insert(0, r'$Root')
    from src.utils.model_setup import onnx_path_for_model
    from src.core.detector import PersonDetector
    import os
    p = onnx_path_for_model('yolo11s')
    if p.is_file():
        d = PersonDetector(p, model_name='yolo11s')
        print('Detector backend:', d.backend_name, '| GPU:', d.using_gpu)
        if not d.using_gpu:
            print('Hint: ONNX CPU fallback often means CUDA 12 runtime DLLs are missing from PATH.')
            print('PATH has cublasLt64_12.dll hint:', any('cuda' in x.lower() for x in os.environ.get('PATH','').split(';')))
    else:
        print('ONNX model missing — run: python scripts/export_onnx.py')
except Exception as e:
    print('Detector test failed:', e)
"@

Write-Host ""
Write-Host "If GPU is False:" -ForegroundColor Yellow
Write-Host "  - CUDA 12 bin on PATH is not enough; add cuDNN 9 bin (folder containing cudnn64_9.dll)." -ForegroundColor Yellow
Write-Host "  - Example: C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64" -ForegroundColor Yellow
Write-Host "  - run.bat prepends cuDNN automatically; or add that folder to your User PATH." -ForegroundColor Yellow
Write-Host "After ultralytics export: pip uninstall -y onnxruntime; pip install onnxruntime-gpu" -ForegroundColor Yellow
