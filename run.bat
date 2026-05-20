@echo off
cd /d "%~dp0"
rem ONNX Runtime CUDA needs cuDNN 9 DLLs on PATH (CUDA toolkit bin is not enough)
if exist "C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64\cudnn64_9.dll" (
  set "PATH=C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64;%PATH%"
) else if exist "C:\Program Files\NVIDIA\CUDNN\v9.11\bin\12.9\cudnn64_9.dll" (
  set "PATH=C:\Program Files\NVIDIA\CUDNN\v9.11\bin\12.9;%PATH%"
)
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe src\main.py
) else (
    python src\main.py
)
if errorlevel 1 (
    echo.
    echo Person Detector exited with an error. See %%APPDATA%%\PersonDetector\person_detector.log
    pause
)
