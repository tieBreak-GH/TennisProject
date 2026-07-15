import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
from tqdm import tqdm

import config
from postprocess import refine_kps
from homography import get_trans_matrix, refer_kps

class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        self.model = BallTrackerNet(out_channels=15)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location='cpu', weights_only=True))
            self.model = self.model.to(device)
            self.model.eval()

    def _infer_frame(self, image, scale_x, scale_y):
        """ Run the court-keypoint model on a single frame.
        :return
            matrix_trans: inverse homography matrix (frame -> court reference), or None
            points: 14 reference keypoints projected into this frame, or None
        """
        output_width = 640
        output_height = 360
        img = cv2.resize(image, (output_width, output_height))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        with torch.no_grad():
            out = self.model(inp.float().to(self.device))[0]
        pred = torch.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            ret, heatmap = cv2.threshold(heatmap, config.COURT_HEATMAP_THRESHOLD, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=config.COURT_HOUGH_MIN_DIST,
                                       param1=config.COURT_HOUGH_PARAM1, param2=config.COURT_HOUGH_PARAM2,
                                       minRadius=config.COURT_HOUGH_MIN_RADIUS, maxRadius=config.COURT_HOUGH_MAX_RADIUS)
            if circles is not None:
                x_pred = circles[0][0][0]*scale_x
                y_pred = circles[0][0][1]*scale_y
                if kps_num not in [8, 12, 9]:
                    x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=config.REFINE_KPS_CROP_SIZE)
                points.append((x_pred, y_pred))
            else:
                points.append(None)

        matrix_trans = get_trans_matrix(points)
        points_res = None
        if matrix_trans is not None:
            points_res = cv2.perspectiveTransform(refer_kps, matrix_trans)
            matrix_trans = cv2.invert(matrix_trans)[1]
        return matrix_trans, points_res

    def infer_model(self, frames, scenes=None, max_probe_frames=config.COURT_MAX_PROBE_FRAMES):
        """ Detect the court homography, reusing one result per scene.

        The camera is assumed static within a scene, so the homography barely
        changes there - probing a handful of frames per scene and broadcasting
        the first valid result avoids running the CNN + Hough/homography math
        on every single frame.
        :params
            frames: list of original video frames
            scenes: list of [start, end) frame-index ranges, or None to treat
                the whole video as one scene
            max_probe_frames: how many frames from the start of a scene to try
                before giving up on that scene (in case the court isn't visible
                in the very first frame(s))
        :return
            matrixes_res, kps_res: same shape/format as before - one entry per
                frame, broadcast from the scene's probed result
        """
        if scenes is None:
            scenes = [[0, len(frames)]]

        orig_height, orig_width = frames[0].shape[:2]
        scale_x = orig_width / 640
        scale_y = orig_height / 360

        matrixes_res = [None] * len(frames)
        kps_res = [None] * len(frames)

        for start, end in tqdm(scenes):
            matrix_trans, points = None, None
            for num_frame in range(start, min(end, start + max_probe_frames)):
                matrix_trans, points = self._infer_frame(frames[num_frame], scale_x, scale_y)
                if matrix_trans is not None:
                    break
            for i in range(start, end):
                matrixes_res[i] = matrix_trans
                kps_res[i] = points

        return matrixes_res, kps_res
