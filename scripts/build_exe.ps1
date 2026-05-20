# Build PersonDetector.exe (one-folder, no console window)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== Person Detector build ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y onnxruntime 2>$null
pip install onnxruntime-gpu --force-reinstall

Write-Host "Exporting ONNX model (if missing)..."
python scripts/export_onnx.py --model yolo11s

if (-not (Test-Path "assets")) {
    New-Item -ItemType Directory -Path "assets" | Out-Null
}
if (-not (Test-Path "assets\icon.ico")) {
    Write-Host "Generating placeholder icon..."
    python scripts/make_icon.py
}

Write-Host "Running PyInstaller..."
pyinstaller build/person_detector.spec --noconfirm

$OutDir = Join-Path $Root "dist\PersonDetector"
Write-Host ""
Write-Host "Build complete: $OutDir\PersonDetector.exe" -ForegroundColor Green
Write-Host "Copy the entire PersonDetector folder to run on another PC."
