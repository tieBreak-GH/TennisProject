import cv2
import numpy as np
import pandas as pd
from collections import deque
from scipy.spatial import distance


def _transform_point(point, matrix):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt_trans = cv2.perspectiveTransform(pt, matrix)
    return pt_trans[0, 0]


def get_ball_speed(ball_track, homography_matrices, fps, window=5, max_speed_kmh=300, smooth_window=5):
    """
    Estimate ball speed (km/h) per frame from court-plane displacement.
    Pixel distance alone is misleading under perspective, so each ball point is
    projected through its frame's homography into real court coordinates
    (CourtReference units are centimeters) before measuring distance.
    :params
        ball_track: list of (x, y) ball pixel coordinates per frame
        homography_matrices: list of image->court homography matrices per frame
        fps: video frame rate
        window: max frame gap used as the speed baseline
        max_speed_kmh: outlier guard, speeds above this are dropped
        smooth_window: rolling median window applied to reduce jitter
    :return
        list of speed values (float or None) aligned with ball_track
    """
    n = len(ball_track)
    raw_speeds = [None] * n
    recent = deque()

    for i in range(n):
        if ball_track[i][0] is None or homography_matrices[i] is None:
            continue
        while recent and i - recent[0] > window:
            recent.popleft()
        if recent:
            j = recent[0]
            pt_i = _transform_point(ball_track[i], homography_matrices[i])
            pt_j = _transform_point(ball_track[j], homography_matrices[j])
            dist_cm = distance.euclidean(pt_i, pt_j)
            dt = (i - j) / fps
            speed_kmh = (dist_cm / 100) / dt * 3.6
            if speed_kmh <= max_speed_kmh:
                raw_speeds[i] = speed_kmh
        recent.append(i)

    smoothed = pd.Series(raw_speeds).rolling(smooth_window, min_periods=1, center=True).median()
    return [None if pd.isna(x) else float(x) for x in smoothed]
