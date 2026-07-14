# TennisProject
Tennis analysis using deep learning and machine learning. <br>
You can check this blog post https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee for more details

![](pics/hard.gif)
![](pics/grass.gif)
![](pics/clay.gif)

### Ball detection
TrackNet was used for detecting tennis ball during the game. For more information you can check this repository: https://github.com/yastrebksv/TrackNet. There you can find 
pretrained weights to check the model.

### Bounce detection
CatBoostRegressor was used to predict ball's bounces during the game based on ball trajectory detected in the previous step. You can check this pretrained model: https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view?usp=drive_link 

### Court detection
It was used neural network for detection 14 points of tennis court. For more information you can check this repository: https://github.com/yastrebksv/TennisCourtDetector. There you can find pretrained weights to check the model.

### Ball speed
Each ball position is projected into real-world court coordinates (cm) via the court homography, giving an instantaneous speed (km/h) drawn next to the ball, plus a stable "peak speed of the current shot" HUD (bounce-to-bounce) that stays readable while the ball is moving fast. Requires the court to be visible/tracked in frame; see `speed_estimator.py`.

### Web UI
Run `streamlit run app.py` for a browser-based UI: upload a video, optionally skip player detection for a faster run, and download the annotated result with ball speed / bounce / court overlay.

For a one-click launch (creates/activates the virtual environment, installs requirements on first run, and opens the app in your browser), use the script for your OS:
- **macOS**: double-click `run_app_mac.command` (or run it from a terminal)
- **Linux**: `./run_app_linux.sh`
- **Windows**: double-click `run_app_windows.bat`

### GPU support
- **NVIDIA (CUDA)**: works out of the box, no setup beyond a normal `pip install` with matching NVIDIA drivers.
- **AMD on Linux (ROCm)**: works out of the box if you install a ROCm build of PyTorch (see https://pytorch.org for the right `--index-url`) — no code changes needed.
- **Apple Silicon (MPS)**: used automatically for ball/court/person detection when available.
- **AMD/Intel/NVIDIA on Windows (DirectML)**: optional, unverified — install `pip install torch-directml` yourself and the code will pick it up automatically for all three models. This package is experimental and its last release may not match the `torch` version pinned in `requirements.txt`, so it isn't installed by default; test compatibility in your own environment.
- Player detection uses YOLO (`ultralytics`) and shares the same device selection as ball/court detection — benchmarked at ~11ms/frame on MPS vs ~16ms/frame on CPU (Apple M-series), so no MPS performance pit like the old Faster R-CNN detector had.

### How to run
Requires Python 3.9+.

1. Clone the repository: `git clone https://github.com/tieBreak-GH/TennisProject.git`
2. Run `pip install -r requirements.txt` to install packages required
3. Download the pretrained weights (links above) and place them under `weights/`:
   - `weights/ball_track_model.pt`
   - `weights/court_model.pt`
   - `weights/bounce_model.cbm`
4. Run the pipeline on a video (any resolution — frames are automatically rescaled to match the models):
   ```
   python main.py \
     --path_ball_track_model weights/ball_track_model.pt \
     --path_court_model weights/court_model.pt \
     --path_bounce_model weights/bounce_model.cbm \
     --path_input_video input_video.mp4 \
     --path_output_video output_video.mp4
   ```
   Or use the web UI instead: `streamlit run app.py`.
