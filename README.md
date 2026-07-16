# TennisProject
Tennis video analysis using deep learning and machine learning: ball/court/player tracking, ball speed, rally & serve segmentation, and an estimated in/out call for serves — all through a batch-friendly web UI or CLI. <br>
You can check this blog post https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee for background on the original ball/court/bounce models.

![](pics/hard.gif)
![](pics/grass.gif)
![](pics/clay.gif)

## Features

- **Ball detection**: TrackNet tracks the ball frame by frame. See https://github.com/yastrebksv/TrackNet. Pretrained weights: [ball_track_model.pt](https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing)
- **Court detection**: a CNN detects 14 court keypoints, cached per camera-angle scene for speed. See https://github.com/yastrebksv/TennisCourtDetector. Pretrained weights: [court_model.pt](https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link)
- **Bounce detection**: a CatBoostRegressor predicts bounces from the ball's trajectory. Pretrained weights: [bounce_model.cbm](https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view?usp=drive_link)

- **Player detection**: YOLO11n locates the top/bottom players and projects them onto a minimap.
- **Ball speed**: each ball position is projected into real-world court coordinates (cm) via the court homography, giving an instantaneous speed (km/h) next to the ball plus a stable "peak speed of the current shot" HUD. Requires the court to be visible/tracked in frame. **Accuracy depends on camera height**: this projection assumes the ball is on the ground plane, so a low/close camera (e.g. a 1.5-2.5m tripod) systematically overstates speed — mount the camera as high and as far back as practical (see `docs/mimari_fizik_gpu_degerlendirme.md` §3 for the underlying geometry and a fix already planned in `docs/uygulama_plani.md`).
- **Rally & serve segmentation**: the video is split into rallies and shots from ball-tracking gaps and bounces; each rally's first shot is labeled a serve if it starts near a baseline.
- **Estimated serve line calling**: each serve gets an estimated **in / out / uncertain** call against its target service box (singles lines), derived purely from the existing homography + bounce detection — no extra model. Deliberately scoped to serves only and shown as an *estimate* (an "uncertain" band near the lines), since bounce-frame precision and homography error make a hard verdict unreliable. See `rally_analyzer.py` and `docs/development_plan.md` (Mantık §7) for the geometry and the accuracy caveats.
- **Highlights & dead-time trim**: auto-cut short clips for the fastest shots / longest rallies, and/or a single video with only rally windows (dead time between points removed) — both written inline during rendering, no extra decode pass.
- **CSV/JSON export**: download the rally table as CSV or the full stats dict as JSON.
- **Batch web UI**: upload and process multiple videos in one session, each with its own results, downloads, and a real per-frame progress bar with ETA.

### Performance & memory

The pipeline runs as a two-pass **streaming** design (`main.py`): pass 1 decodes the video once and keeps only lightweight per-frame metadata (ball/court/player coordinates), pass 2 decodes again and renders straight to disk — the full video is never held in memory, so peak RAM stays roughly constant regardless of video length. Player detection uses YOLO instead of the older Faster R-CNN, and court homography is cached per camera-angle scene rather than recomputed every frame. On CUDA GPUs, the ball/player models additionally batch several frames per forward pass (`config.py`); this is skipped on CPU/MPS where it was measured to be *slower*, not faster. See `docs/development_plan.md` (Teknik §1-2) for the full writeup and benchmarks.

## Web UI
Run `streamlit run app.py` for a browser-based UI: upload one or more videos, optionally skip player detection for a faster run, generate highlight clips and/or a dead-time-trimmed video, and download everything (annotated video, rally table as CSV, full stats as JSON) once processing finishes. Upload limit is 1GB per file (`.streamlit/config.toml`).

For a one-click launch (creates/activates the virtual environment, installs requirements on first run, and opens the app in your browser), use the script for your OS:
- **macOS**: double-click `run_app_mac.command` (or run it from a terminal)
- **Linux**: `./run_app_linux.sh`
- **Windows**: double-click `run_app_windows.bat`

## GPU support
- **NVIDIA (CUDA)**: works out of the box, no setup beyond a normal `pip install` with matching NVIDIA drivers. Also the only backend where batch inference (see above) kicks in.
- **AMD on Linux (ROCm)**: works out of the box if you install a ROCm build of PyTorch (see https://pytorch.org for the right `--index-url`) — no code changes needed.
- **Apple Silicon (MPS)**: used automatically for ball/court/person detection when available.
- **AMD/Intel/NVIDIA on Windows (DirectML)**: optional, unverified — install `pip install torch-directml` yourself and the code will pick it up automatically for all three models. This package is experimental and its last release may not match the `torch` version pinned in `requirements.txt`, so it isn't installed by default; test compatibility in your own environment.
- Player detection uses YOLO (`ultralytics`) and shares the same device selection as ball/court detection — benchmarked at ~11ms/frame on MPS vs ~16ms/frame on CPU (Apple M-series), so no MPS performance pit like the old Faster R-CNN detector had.

### AMD Windows GPU setup (e.g. RX 9070 XT)

`pip install -r requirements.txt` on Windows pulls the **CPU-only** PyTorch wheel by default (there's no Windows CUDA/ROCm wheel on PyPI) — the pipeline will silently run on CPU even with a capable AMD card installed. Check what's actually being used:
```bash
python check_gpu.py
```
This prints the detected torch build, which backend `main._select_device` would pick, and a CPU-vs-GPU ms/frame benchmark (no model weights needed). If it reports a CPU-only build, pick one of:

1. **`torch-directml` (easiest)** — works with any DX12-capable AMD/Intel/NVIDIA GPU on Windows natively, no dual-boot or WSL required:
   ```bash
   pip install torch-directml
   ```
   Re-run `python check_gpu.py` to confirm the card is detected. This package is experimental and may pin an older `torch` version than `requirements.txt` (see the bullet above) — recent GPU generations (e.g. RDNA4) aren't guaranteed to be validated against it yet, so verify output correctness against a known-good CPU run before trusting results.
2. **WSL2 + ROCm (better performance, more setup)** — install WSL2 with GPU passthrough, then a ROCm build of PyTorch inside it (`--index-url` per https://pytorch.org; RDNA4/gfx1201 needs ROCm ≥ 6.4). Once installed, the Linux ROCm path above applies unchanged.

No code changes are needed for either path — `main._select_device` already tries DirectML/ROCm/MPS before falling back to CPU. The web UI's "İşlem birimi" (processing device) selector lets you force CPU if an experimental GPU backend misbehaves, without editing any files; the results panel always shows which device actually ran.

Note: GPU acceleration only speeds up **model inference** (ball/court/player detection - the actual bottleneck). Video decode/encode (OpenCV/FFmpeg) always runs on CPU regardless of backend.

## Configuration
Tunable thresholds (confidence scores, Hough parameters, rally-gap/baseline margins, the serve line-calling margin, etc.) live in one place, `config.py`, each documented with what it controls. Pretrained-model input shapes (e.g. the 640×360 ball/court model resolution) are deliberately *not* there — changing those without retraining would silently break the model.

## How to run
Requires Python 3.9+.

1. Clone the repository: `git clone https://github.com/tieBreak-GH/TennisProject.git`
2. Run `pip install -r requirements.txt` to install packages required
3. Download the pretrained weights by running the automated script:
   ```bash
   python download_weights.py
   ```
   Or download them manually and place them under a `weights/` directory:
   - [ball_track_model.pt](https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing)
   - [court_model.pt](https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link)
   - [bounce_model.cbm](https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view?usp=drive_link)
4. Run the pipeline on a video (any resolution — frames are automatically rescaled to match the models):
   ```
   python main.py \
     --path_ball_track_model weights/ball_track_model.pt \
     --path_court_model weights/court_model.pt \
     --path_bounce_model weights/bounce_model.cbm \
     --path_input_video input_video.mp4 \
     --path_output_video output_video.mp4
   ```
   Or use the web UI instead (which will also check and download missing weights automatically): `streamlit run app.py`.


## Tests
Pure-function unit tests (homography solving, line intersection, bounce postprocessing, rally/serve/line-call segmentation) live under `tests/` and don't need model weights or a GPU:
```
pip install -r requirements-dev.txt
pytest
```

## Project docs
(Turkish) `docs/development_plan.md` tracks completed work and the open roadmap, with the reasoning/benchmarks behind each decision; `docs/comparison_report.md` compares this project to similar open-source and commercial tools and covers mobile web usability; `docs/tennis_analysis_report.md` and `docs/code_review_report.md` cover the architecture and a detailed code review in more depth.
