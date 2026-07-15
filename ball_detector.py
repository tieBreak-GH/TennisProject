from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

import config

class BallDetector:
    def __init__(self, path_model=None, device='cuda'):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location='cpu', weights_only=True))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_batch(self, resized_window, prev_pred, scale_x, scale_y):
        """ Run the model forward pass on one batch of sliding 3-frame
        windows, then sequential (prev_pred-dependent) postprocessing.
        Factored out of infer_model so streaming callers (main.py) can feed
        it a bounded window of already-resized frames pulled from a video
        decode loop, instead of requiring the whole video resized in memory.
        :params
            resized_window: list of already-resized (640x360) frames;
                produces len(resized_window)-2 predictions - window j uses
                (resized_window[j], resized_window[j-1], resized_window[j-2])
                as (current, prev, preprev) for j in [2, len(resized_window)-1]
            prev_pred: [x, y] postprocess state carried in from the previous
                batch (sequential outlier-filter dependency)
            scale_x, scale_y: scale factors back to the original frame size
        :return
            (predictions, prev_pred): predictions is a list of (x, y)
            tuples, one per window; prev_pred is the updated postprocess
            state to pass into the next call
        """
        num_windows = len(resized_window) - 2
        batch_inputs = []
        for j in range(2, num_windows + 2):
            imgs = np.concatenate((resized_window[j], resized_window[j-1], resized_window[j-2]), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            batch_inputs.append(imgs)
        inp = np.stack(batch_inputs, axis=0)

        with torch.no_grad():
            out = self.model(torch.from_numpy(inp).float().to(self.device))
        output = out.argmax(dim=1).detach().cpu().numpy()

        predictions = []
        for i in range(output.shape[0]):
            x_pred, y_pred = self.postprocess(output[i:i+1], prev_pred, scale_x, scale_y)
            prev_pred = [x_pred, y_pred]
            predictions.append((x_pred, y_pred))
        return predictions, prev_pred

    def infer_model(self, frames, batch_size=config.BALL_INFER_BATCH_SIZE):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
            batch_size: how many 3-frame windows to run through the model
                forward pass at once. Postprocessing stays per-frame and
                sequential (each frame's outlier filter depends on the
                previous frame's prediction), so only the GPU-bound forward
                pass is batched; results are unaffected by batch_size since
                the model is in eval() mode (no batch-dependent layers).
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        orig_height, orig_width = frames[0].shape[:2]
        scale_x = orig_width / self.width
        scale_y = orig_height / self.height

        resized = [cv2.resize(f, (self.width, self.height)) for f in frames]

        num_windows = len(frames) - 2
        for batch_start in tqdm(range(0, num_windows, batch_size)):
            batch_end = min(batch_start + batch_size, num_windows)
            predictions, prev_pred = self.infer_batch(resized[batch_start:batch_end+2], prev_pred, scale_x, scale_y)
            ball_track.extend(predictions)
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale_x=2, scale_y=2, max_dist=config.BALL_MAX_JUMP_PX):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale_x, scale_y: scale factors for conversion back to the original frame size
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, config.BALL_HEATMAP_THRESHOLD, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=config.BALL_HOUGH_MIN_DIST,
                                   param1=config.BALL_HOUGH_PARAM1, param2=config.BALL_HOUGH_PARAM2,
                                   minRadius=config.BALL_HOUGH_MIN_RADIUS, maxRadius=config.BALL_HOUGH_MAX_RADIUS)
        x, y = None, None
        if circles is not None:
            if prev_pred[0] is not None:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale_x
                    y_temp = circles[0][i][1]*scale_y
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0]*scale_x
                y = circles[0][0][1]*scale_y
        return x, y
