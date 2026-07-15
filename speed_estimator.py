import cv2
import numpy as np
import pandas as pd
from collections import deque
from scipy.spatial import distance

import config


def _transform_point(point, matrix):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt_trans = cv2.perspectiveTransform(pt, matrix)
    return pt_trans[0, 0]


def get_ball_speed(ball_track, homography_matrices, fps, window=config.BALL_SPEED_WINDOW_FRAMES,
                    max_speed_kmh=config.BALL_SPEED_MAX_KMH, smooth_window=config.BALL_SPEED_SMOOTH_WINDOW):
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


def get_shot_max_speed(ball_speed, bounce_frames):
    """
    Per-frame instantaneous speed changes too fast to read while the ball is moving.
    This instead gives each bounce-to-bounce flight segment a single value: the peak
    speed reached during that segment (i.e. the shot that produced it), so a fixed
    on-screen HUD can hold a stable, readable number for the whole flight.
    :params
        ball_speed: list of speed values (km/h) per frame, as returned by get_ball_speed
        bounce_frames: iterable of frame indices where the ball bounces
    :return
        list aligned with ball_speed: each frame holds its segment's peak speed (or None)
    """
    n = len(ball_speed)
    boundaries = sorted(set(bounce_frames))

    segments = []
    start = 0
    for b in boundaries:
        if 0 <= b <= n and b > start:
            segments.append((start, b))
        start = b
    segments.append((start, n))

    shot_max_speed = [None] * n
    for seg_start, seg_end in segments:
        seg_speeds = [s for s in ball_speed[seg_start:seg_end] if s is not None]
        seg_max = max(seg_speeds) if seg_speeds else None
        for i in range(seg_start, seg_end):
            shot_max_speed[i] = seg_max
    return shot_max_speed
