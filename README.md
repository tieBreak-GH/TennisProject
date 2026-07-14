# TennisProject
Tennis analysis using deep learning and machine learning. <br>
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

### GPU support
- **NVIDIA (CUDA)**: works out of the box, no setup beyond a normal `pip install` with matching NVIDIA drivers.
- **AMD on Linux (ROCm)**: works out of the box if you install a ROCm build of PyTorch (see https://pytorch.org for the right `--index-url`) — no code changes needed.
- **Apple Silicon (MPS)**: used automatically for ball/court detection when available.
- **AMD/Intel/NVIDIA on Windows (DirectML)**: optional, unverified — install `pip install torch-directml` yourself and the code will pick it up automatically for ball/court detection. This package is experimental and its last release may not match the `torch` version pinned in `requirements.txt`, so it isn't installed by default; test compatibility in your own environment.
- Player detection (Faster R-CNN) always stays on CPU/CUDA — MPS measured ~67x slower for this model, and DirectML is assumed to have the same risk until proven otherwise.

### How to run
Prepare a video file with resolution 1280x720
1. Clone the repository `https://github.com/yastrebksv/TennisProject.git`
2. Run `pip install -r requirements.txt` to install packages required
3. Run `python main.py <args>`

   

