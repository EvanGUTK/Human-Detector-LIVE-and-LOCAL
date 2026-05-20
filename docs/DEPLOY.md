# Deployment guide

## Push to GitHub (maintainers)

```powershell
cd "Human-Detector-LIVE-and-LOCAL"
git init
git add .
git commit -m "Initial public release: AI Testing Lab v4"
git branch -M main
git remote add origin https://github.com/EvanGUTK/Human-Detector-LIVE-and-LOCAL.git
git pull origin main --allow-unrelated-histories
# Resolve LICENSE if prompted (keep MIT), then:
git push -u origin main
```

If the remote only contains an empty LICENSE commit and you intend to replace it:

```powershell
git push -u origin main --force
```

## Clone and run (development)

```powershell
git clone https://github.com/EvanGUTK/Human-Detector-LIVE-and-LOCAL.git
cd Human-Detector-LIVE-and-LOCAL
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\run.bat
```

## GPU (Windows)

1. NVIDIA driver + CUDA 12 runtime on `PATH`
2. cuDNN 9 bin on `PATH` (see README)
3. `.\scripts\check_gpu.ps1` → `GPU: True`

## Models

| Family | First run |
|--------|-----------|
| YOLO11 / YOLOv8 | Auto-export via Ultralytics to `models/` |
| RT-DETR-s | Auto-export via Ultralytics |
| PeopleNet / DetectNet_v2 | NGC download if `NGC_API_KEY` set, or manual ONNX |

### TAO models (NGC)

```powershell
setx NGC_API_KEY "your-ngc-key"
# Install NGC CLI from NVIDIA
ngc registry model list nvidia/tao/peoplenet
```

Cached under `models/ngc/<model_id>/`, symlinked to `models/<model_id>_tao.onnx`.

Manual fallback: place ONNX at `models/peoplenet_tao.onnx` or `models/detectnet_v2_tao.onnx`.

### Verify

```powershell
.\.venv\Scripts\python scripts\verify_models.py
```

## GitHub Actions

CI runs `py_compile` and `verify_models.py --skip-gpu` (no GPU on runner).

## Release build

```powershell
.\scripts\build_exe.ps1
```

Ship `dist/PersonDetector/` — users still download models on first run.

## Secrets

Do **not** commit `NGC_API_KEY`, `.env`, or `models/*.onnx`.
