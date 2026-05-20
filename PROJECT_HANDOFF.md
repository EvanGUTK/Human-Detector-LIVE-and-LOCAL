# Person Detector — AI Testing Lab (handoff)

**Paste into a new chat:**

> Read `PROJECT_HANDOFF.md` in the Person Detector repo. Do not edit `*.plan.md` files unless I ask.

---

## What this is

**Person Detector — AI Testing Lab**: local Windows app for **CCTV/desktop testing**, **multi-class COCO detection** (configurable subset), **zones/alerts** (configurable class IDs), **face ID** (configurable class IDs), optional **fire/smoke overlay** (second ONNX), and a full **annotate → train → validate → compare** loop (YOLO11 / YOLOv8 / RT-DETR / TAO-style ONNX).

**Path:** `e:\Cursor Projects\Person Detector`

**Plans (reference):** v2 `person_detector_v2_3757f215.plan.md`, v3 `ai_testing_lab_v3_be33f5b0.plan.md`

---

## Quick run

```powershell
cd "e:\Cursor Projects\Person Detector"
.\run.bat
.\scripts\check_gpu.ps1
pip install -r requirements.txt   # includes mss, insightface, pyqtgraph
```

**User data:** `%APPDATA%\PersonDetector\` — `config.json`, `profiles/`, faces, **`lab/projects/`** (annotations + training runs), **`models_registry.json`**

**Models on disk:** repo `models/` (or next to exe when frozen) — ONNX naming: see [ONNX / inference size](#onnx--inference-size) below.

---

## Implemented feature map

| Feature | Where |
|---------|--------|
| **COCO class set** | `detect_class_ids` (80-class Ultralytics order); Settings → **COCO classes…** (`src/ui/coco_class_picker.py`) |
| **CCTV preset** | Settings → **CCTV preset** — traffic class IDs, YOLO11m, imgsz 960, lower vehicle thresholds (`CCTV_TRAFFIC_CLASS_IDS` in `src/utils/coco_classes.py`) |
| **Per-class confidence** | `confidence_per_class` dict keyed by COCO **name**; Settings table for selected classes |
| **Inference size** | `model_imgsz` 416 / 640 / 960 / 1280; separate ONNX paths (`src/utils/model_setup.py`) |
| **Faded below-threshold boxes** | `show_faded_low_conf`; primary path still uses thresholds; gray overlay for debug (`YoloDetector.last_faded_detections`, `pipeline.py`) |
| **Fire/smoke** | Optional second ONNX; **overlay only**, no zone enter (`src/core/fire_smoke_detector.py`, pipeline merge) |
| **Import ONNX (NGC/BYO)** | Settings → **Import ONNX…** → `ModelRegistry.register_custom` (must match YOLO-style ONNX head used elsewhere) |
| **Video transport** | `VideoTransportBar` + `VideoFilePlayback` — file pause/seek/speed separate from detection pause (`video_transport.py`, `file_playback.py`, Monitor toolbar) |
| **Model Health** | Lab tab: test/download/export ONNX (`model_health_panel.py`, `scripts/verify_models.py`) |
| **TAO models** | Distinct NGC resources for PeopleNet vs DetectNet_v2; family decoders (`ngc_download.py`, `tao_decoders.py`) |
| **Lab dashboard** | Test clips under `%APPDATA%/PersonDetector/clips/` (`lab_dashboard.py`) |
| **Compare controls** | Shared transport, family filter, latency ms, overlay PNG export (`compare_panel.py`) |
| **Compare desktop mode** | Compare source can be **Video file** or **Desktop** (desktop pause = freeze-only) (`src/ui/compare_panel.py`) |
| **Zones delete UX** | Select zone → Delete button or **Delete** key (`src/ui/zones_panel.py`) |
| **Quick person/car** | Settings checkboxes sync with classes **0** and **2** in `detect_class_ids` |
| **Desktop capture** | Toolbar + `screen_capture.py` (monitor + ROI) |
| **Train / Compare / Evaluate** | `train_panel`, `compare_panel`, `evaluate_panel` |
| **Built-in model IDs** | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`, `yolov8m`, `rtdetr-s`, `peoplenet`, `detectnet_v2` + custom / imported (`src/lab/model_registry.py`) |
| **v2 baseline** | Monitor, Zones, Analytics, Faces, profiles, overlay, file replay, debug `D` |

---

## Tabs

| Tab | Use |
|-----|-----|
| Monitor | Live: webcam, file, **screen**, detection, zones; file mode shows **video transport** (pause video ≠ pause detection) |
| **Lab** | Dashboard → Compare → Train → Evaluate → Model Health |
| Train | Annotate (COCO class dropdown) → YOLO export / train → custom ONNX; **Play** auto-advance frames |
| Compare | Two models on same source (**video or desktop**), shared transport for file mode |
| Evaluate | mAP, label IoU, batch folder; **model picker** |
| Zones / Analytics / Faces | Same as v2 |
| **Settings** | Model, **imgsz**, COCO classes, thresholds, fire/smoke, camera, alerts, face, display |

---

## Training workflow

1. **Train** → New project → attach video  
2. Scrub frames → **Class** (COCO dropdown) → **Draw box** → repeat  
3. **Auto train/val split** → **Train model** (Ultralytics, background)  
4. Model appears under **Settings → Model** (custom id)  
5. **Compare** / **Evaluate** as needed  

**Storage:** `%APPDATA%\PersonDetector\lab\projects\<id>/` — `project.json`, `annotations.json`, `exports/yolo/`, `runs/`

**Export:** `src/lab/yolo_exporter.py` builds `data.yaml` with **`nc`** and **`names`** reindexed **0..K-1** from whatever class strings appear in annotations. Legacy **person+car-only** projects still map to two YOLO ids.

**Custom `custom_*` models:** `YoloDetector._remap_custom_classes()` maps YOLO class **1 → COCO car (2)** when `model_name` starts with `custom_`.

---

## Detection behavior

- **Zones & alerts:** class-configurable via `zone_alert_class_ids` (default `[0]`) — zone enter/exit uses track feet point.  
- **Face recognition:** class-configurable via `face_rec_class_ids` (default `[0]`), with stride-aware throttling to reduce unknown-track overhead.  
- **Other COCO classes:** tracked and drawn; **no** zone policy (vehicles use vehicle-ish colors in `pipeline._draw`).  
- **Shared detector kwargs:** `yolo_kwargs_from_config(cfg)` in `src/core/detector.py` — used by `main.py`, `main_window._recreate_detector`, Compare, Evaluate, `scripts/benchmark.py`.

---

## ONNX / inference size

- **`onnx_path_for_model(name, imgsz)`** (`src/utils/model_setup.py`):  
  - **`imgsz == 640`:** if `models/{name}.onnx` exists (legacy), use it; else `models/{name}_640.onnx`.  
  - **Other imgsz:** `models/{name}_{imgsz}.onnx`.  
- **`ensure_onnx_model(name, imgsz)`** exports via Ultralytics and moves the produced `.onnx` to that path (handles nested export folder).

Changing **imgsz** without changing model id still requires a **different file** on disk — do not mix a 640-exported graph with 1280 letterbox expectations.

---

## Config (representative)

Full defaults also live in `src/utils/config.py` `_DEFAULTS` and `config/default.yaml`. On load, **`migrate_config_inplace`** fills missing keys; **`sync_legacy_detect_flags`** keeps `detect_person` / `detect_car` aligned with `detect_class_ids`.

```yaml
# Detection
detect_class_ids: [0]                 # list of COCO 0..79 (non-empty after migrate)
zone_alert_class_ids: [0]             # which classes can trigger zone enter/exit
face_rec_class_ids: [0]               # which classes are passed to face ID
detect_person: true                   # legacy UI; synced from ids 0 / 2
detect_car: false
confidence: 0.45                      # default when no per-class override
confidence_per_class: {}              # e.g. {"car": 0.28, "truck": 0.28}
model_imgsz: 640
active_model_id: yolo11s
show_faded_low_conf: false

# Fire/smoke (overlay only)
fire_smoke_enabled: false
fire_smoke_onnx_path: ""
fire_smoke_full_frame: true

# Input / screen
input_source: webcam                  # webcam | file | screen
screen_monitor: 1
screen_region: null                   # or [x, y, w, h] on chosen monitor
screen_fps_cap: 0
infer_max_width: 1280
infer_max_height: 720
preview_mode: performance
model_preset: balanced
compare_source: video                 # video | desktop (Compare tab default)
compare_playback_speed: 1.0           # Compare file playback speed
```

**Profiles:** keys saved per room are listed in **`PROFILE_KEYS`** in `src/core/profile_manager.py`. **`apply_to_config`** backfills `detect_class_ids` from legacy `detect_person`/`detect_car` when an old profile JSON omits the new field.

---

## Key files

| Area | Path |
|------|------|
| Detector + `yolo_kwargs_from_config` | `src/core/detector.py` |
| Pipeline, draw, faded, fire merge | `src/core/pipeline.py` |
| Fire/smoke ONNX stub | `src/core/fire_smoke_detector.py` |
| Tracker (optional `display_name` on tracks) | `src/core/tracker.py` |
| COCO names + CCTV preset IDs | `src/utils/coco_classes.py` |
| Config load/migrate | `src/utils/config.py` |
| ONNX paths / export | `src/utils/model_setup.py` |
| Built-in + custom models | `src/lab/model_registry.py` |
| YOLO dataset export | `src/lab/yolo_exporter.py` |
| Settings UI | `src/ui/settings_panel.py` |
| Compare UI + controls | `src/ui/compare_panel.py` |
| COCO picker dialog | `src/ui/coco_class_picker.py` |
| Zones UX | `src/ui/zones_panel.py` |
| App shell | `src/ui/main_window.py` |
| Screen capture | `src/core/screen_capture.py` |
| Frozen bundle | `build/person_detector.spec` (includes `coco_class_picker`, `coco_classes`, `fire_smoke_detector`) |

---

## GPU / training

- Inference: **ONNX CUDA** → **PyTorch CUDA** → **CPU** (see README GPU section).  
- ORT wheels still require CUDA 12 runtime DLLs even if you have newer driver/toolkit components installed.
- Common Windows fix: CUDA 12 `bin` on PATH is not enough — add cuDNN 9 `bin` (e.g. `C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64` so `cudnn64_9.dll` resolves). `run.bat` prepends it; validate with `scripts/check_gpu.ps1`.
- Training: Ultralytics + PyTorch; first run can download weights; long runs better in venv than frozen exe only.

---

## Deferred (v4+)

RTSP, auto-label, TensorRT EP, webhooks, clip-on-alert, **zone rules for fire/smoke** (currently overlay-only by design).

---

*Last updated: handoff refresh (compare controls + desktop source, configurable zone/face classes, GPU troubleshooting note).*
