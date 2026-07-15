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

    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        orig_height, orig_width = frames[0].shape[:2]
        scale_x = orig_width / self.width
        scale_y = orig_height / self.height

        # each frame is resized once and reused as "prev"/"preprev" in the next
        # two iterations, instead of being resized 3 times across the window
        resized_preprev = cv2.resize(frames[0], (self.width, self.height))
        resized_prev = cv2.resize(frames[1], (self.width, self.height))

        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (self.width, self.height))
            imgs = np.concatenate((img, resized_prev, resized_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred, scale_x, scale_y)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))

            resized_preprev = resized_prev
            resized_prev = img
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
