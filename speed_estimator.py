import cv2
import numpy as np
import pandas as pd
from collections import deque
from scipy.spatial import distance

import config
from camera_calib import estimate_focal, decompose_pose
from trajectory_3d import fit_segment_trajectory, segment_speed_series, is_reliable_fit


def _transform_point(point, matrix):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt_trans = cv2.perspectiveTransform(pt, matrix)
    return pt_trans[0, 0]


def _segment_start_per_frame(n, bounce_frames):
    """
    For each frame, the start of its bounce-to-bounce segment (same boundary
    semantics as get_shot_max_speed/rally_analyzer.segment_shots: a bounce
    frame belongs to the segment it starts, not the one it ends).
    :return
        list of length n, segment_start[i] <= i
    """
    boundaries = sorted(b for b in set(bounce_frames) if 0 <= b < n)
    starts = [0] * n
    seg_start = 0
    b_idx = 0
    for i in range(n):
        if b_idx < len(boundaries) and boundaries[b_idx] == i:
            seg_start = i
            b_idx += 1
        starts[i] = seg_start
    return starts


def get_ball_speed(ball_track, homography_matrices, fps, bounce_frames=(), window=config.BALL_SPEED_WINDOW_FRAMES,
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
        bounce_frames: iterable of frame indices where the ball bounces. The
            speed window baseline never reaches past a bounce into the
            previous shot - otherwise the straight-line (chord) distance
            between a pre-bounce and post-bounce point undershoots the
            ball's actual (bent) path, understating speed right around the
            bounce/hit.
        window: max frame gap used as the speed baseline
        max_speed_kmh: outlier guard, speeds above this are dropped
        smooth_window: rolling median window applied to reduce jitter
    :return
        list of speed values (float or None) aligned with ball_track
    """
    n = len(ball_track)
    raw_speeds = [None] * n
    recent = deque()
    segment_start = _segment_start_per_frame(n, bounce_frames)

    for i in range(n):
        if ball_track[i][0] is None or homography_matrices[i] is None:
            continue
        while recent and (i - recent[0] > window or recent[0] < segment_start[i]):
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


def _bounce_segments(n, bounce_frames):
    """
    (start, end) bounce-to-bounce segments partitioning [0, n) - same boundary
    semantics as _segment_start_per_frame/get_shot_max_speed (a bounce frame
    starts the next segment, not ends the current one).
    """
    boundaries = sorted(b for b in set(bounce_frames) if 0 <= b <= n)
    segments = []
    start = 0
    for b in boundaries:
        if b > start:
            segments.append((start, b))
        start = b
    segments.append((start, n))
    return segments


def get_ball_speed_3d(ball_track, homography_matrices, scenes, fps, bounce_frames, image_size):
    """
    Camera-height-independent per-frame ball speed (km/h), fit from the
    ball's actual 3D flight path instead of speed_estimator.get_ball_speed's
    ground-plane projection (see docs/mimari_fizik_gpu_degerlendirme.md §3
    for why the latter is systematically wrong for an airborne point, by an
    amount that grows as the camera gets lower).

    For each scene, the court homography (constant per scene - same
    assumption analyze_streaming already makes) yields a camera calibration
    (camera_calib.estimate_focal + decompose_pose) once; each bounce-to-bounce
    segment within that scene is then fit independently
    (trajectory_3d.fit_segment_trajectory) from its ball pixel detections.
    A segment's result is used only where trajectory_3d.is_reliable_fit says
    so - otherwise those frames are left None so the caller (estimate_ball_speed)
    falls back to the 2D method for them.
    :params
        ball_track: list of (x, y) ball pixel coordinates per frame
        homography_matrices: list of image->court homography matrices per
            frame (None where no court was found), as returned by
            main.analyze_streaming - constant within each scene
        scenes: list of (start, end) frame ranges, as returned by
            main.analyze_streaming
        fps: video frame rate
        bounce_frames: iterable of frame indices where the ball bounces
        image_size: (width, height) of the video frames, in pixels
    :return
        (speeds_3d, camera_height_cm): both lists aligned with ball_track.
        speeds_3d[i] is the 3D-fit speed (km/h) at frame i, or None where no
        reliable fit was available. camera_height_cm[i] is that frame's
        scene's recovered camera height (cm), or None if calibration failed
        for that scene - informational (e.g. for a UI caption), independent
        of whether any segment in that scene had a reliable fit.
    """
    n = len(ball_track)
    speeds_3d = [None] * n
    camera_height_cm = [None] * n
    if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
        return speeds_3d, camera_height_cm
    segments = _bounce_segments(n, bounce_frames)

    for scene_start, scene_end in scenes:
        scene_end = min(scene_end, n)
        if scene_end <= scene_start:
            continue
        H_i2c = homography_matrices[scene_start]
        if H_i2c is None:
            continue
        H_c2i = np.linalg.inv(H_i2c)
        focal = estimate_focal(H_c2i, image_size)
        if focal is None:
            continue
        K = np.array([[focal, 0, image_size[0] / 2],
                      [0, focal, image_size[1] / 2],
                      [0, 0, 1]])
        R, t, C = decompose_pose(H_c2i, K)
        if R is None:
            continue

        height_cm = float(C[2])
        for i in range(scene_start, scene_end):
            camera_height_cm[i] = height_cm

        for seg_start, seg_end in segments:
            s, e = max(seg_start, scene_start), min(seg_end, scene_end)
            if e - s < 3:
                continue
            idxs = [i for i in range(s, e) if ball_track[i][0] is not None]
            if len(idxs) < 3:
                continue

            times = np.array([i / fps for i in idxs])
            pixel_coords = np.array([ball_track[i] for i in idxs])
            fit = fit_segment_trajectory(times, pixel_coords, K, R, t, C)
            if not is_reliable_fit(fit):
                continue

            speeds = segment_speed_series(fit['v0'], times, t0=times[0])
            for idx, speed in zip(idxs, speeds):
                speeds_3d[idx] = float(speed)

    return speeds_3d, camera_height_cm


def estimate_ball_speed(ball_track, homography_matrices, scenes, fps, bounce_frames, image_size,
                         method='auto', window=config.BALL_SPEED_WINDOW_FRAMES,
                         max_speed_kmh=config.BALL_SPEED_MAX_KMH, smooth_window=config.BALL_SPEED_SMOOTH_WINDOW):
    """
    Dispatcher: per-frame ball speed (km/h) from the 3D method where
    reliable, falling back to the legacy 2D ground-projection method
    (get_ball_speed) everywhere else. Both methods end up bounded by a
    physical speed ceiling (2D via its own max_speed_kmh; 3D via
    trajectory_3d.is_reliable_fit's max_speed_kmh gate) - real broadcast
    footage (Faz 5.1) showed the 3D fit's covariance-based confidence alone
    is not enough to catch every bad fit (see is_reliable_fit's docstring),
    so a plausibility ceiling is required for it too, not just 2D's
    ground-projection magnification error.
    :params
        method: 'auto' (3D where reliable, 2D fallback elsewhere - the
            normal mode), '2d' (force the legacy method everywhere), '3d'
            (3D only - frames without a reliable fit are left None, no 2D
            fallback; mainly useful for the height-invariance validation)
        (other params: see get_ball_speed / get_ball_speed_3d)
    :return
        (speeds, method_per_frame, camera_height_cm): all lists aligned with
        ball_track. method_per_frame[i] is '3d', '2d', or None (no speed
        available). camera_height_cm as returned by get_ball_speed_3d.
    """
    n = len(ball_track)

    speeds_3d, camera_height_cm = (
        get_ball_speed_3d(ball_track, homography_matrices, scenes, fps, bounce_frames, image_size)
        if method != '2d' else ([None] * n, [None] * n))
    speeds_2d = (
        get_ball_speed(ball_track, homography_matrices, fps, bounce_frames,
                        window=window, max_speed_kmh=max_speed_kmh, smooth_window=smooth_window)
        if method != '3d' else [None] * n)

    speeds = [None] * n
    method_per_frame = [None] * n
    for i in range(n):
        if speeds_3d[i] is not None:
            speeds[i] = speeds_3d[i]
            method_per_frame[i] = '3d'
        elif speeds_2d[i] is not None:
            speeds[i] = speeds_2d[i]
            method_per_frame[i] = '2d'

    return speeds, method_per_frame, camera_height_cm


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
