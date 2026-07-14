import cv2
import numpy as np

from court_reference import CourtReference

_court_ref = CourtReference()
_BASELINE_TOP_Y = _court_ref.baseline_top[0][1]
_BASELINE_BOTTOM_Y = _court_ref.baseline_bottom[0][1]


def _to_court_point(point, matrix):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt_trans = cv2.perspectiveTransform(pt, matrix)
    return pt_trans[0, 0]


def _is_near_baseline(court_point, margin_cm=300):
    """
    Whether a court-plane point (CourtReference units, ~cm) falls within
    margin_cm of either baseline - used to sanity-check a serve candidate's
    starting position.
    """
    y = court_point[1]
    return abs(y - _BASELINE_TOP_Y) <= margin_cm or abs(y - _BASELINE_BOTTOM_Y) <= margin_cm


def segment_rallies(ball_track, fps, max_gap_seconds=1.5):
    """
    Split the video into rallies based on gaps in continuous ball tracking.
    A gap longer than max_gap_seconds (no ball detected) is treated as the
    boundary between two points - this is an approximate default, not
    calibrated against real footage, and may need tuning per video.
    :params
        ball_track: list of (x, y) ball pixel coordinates per frame
        fps: video frame rate
        max_gap_seconds: minimum gap (in seconds) between detections to
            start a new rally
    :return
        list of (start_frame, end_frame) rally windows (end exclusive)
    """
    n = len(ball_track)
    max_gap_frames = max(1, int(round(max_gap_seconds * fps)))

    tracked = [i for i in range(n) if ball_track[i][0] is not None]
    if not tracked:
        return []

    rallies = []
    start = tracked[0]
    prev = tracked[0]
    for i in tracked[1:]:
        if i - prev > max_gap_frames:
            rallies.append((start, prev + 1))
            start = i
        prev = i
    rallies.append((start, prev + 1))
    return rallies


def segment_shots(start, end, bounce_frames):
    """
    Split a rally window into shot windows using bounce frames as boundaries
    (same segmentation pattern as speed_estimator.get_shot_max_speed).
    :params
        start, end: rally window (end exclusive)
        bounce_frames: iterable of frame indices where the ball bounces
    :return
        list of (start_frame, end_frame) shot windows within the rally
    """
    boundaries = sorted(b for b in set(bounce_frames) if start < b < end)

    shots = []
    seg_start = start
    for b in boundaries:
        shots.append((seg_start, b))
        seg_start = b
    shots.append((seg_start, end))
    return shots


def analyze_rallies(ball_track, bounces, homography_matrices, ball_speed, fps,
                     max_gap_seconds=1.5, baseline_margin_cm=300):
    """
    Segment the video into rallies and shots, and label each rally's first
    shot as a serve if its starting ball position projects near a baseline.
    This is the main defense against clips that start mid-rally: the first
    shot of the video trivially looks like "after a gap", but if it doesn't
    start near a baseline it is not labeled a serve.
    :params
        ball_track: list of (x, y) ball pixel coordinates per frame
        bounces: iterable of frame indices where the ball bounces
        homography_matrices: list of image->court homography matrices per frame
        ball_speed: list of speed values (km/h) per frame, as returned by
            speed_estimator.get_ball_speed
        fps: video frame rate
        max_gap_seconds: see segment_rallies
        baseline_margin_cm: see _is_near_baseline
    :return
        list of dicts, one per rally:
        {rally_no, start_frame, end_frame, duration_s, num_shots,
         shots: [{shot_no, start_frame, end_frame, is_serve}],
         avg_speed_kmh, max_speed_kmh}
    """
    rally_windows = segment_rallies(ball_track, fps, max_gap_seconds)
    bounce_set = set(bounces)

    rallies = []
    for rally_no, (r_start, r_end) in enumerate(rally_windows, start=1):
        shot_windows = segment_shots(r_start, r_end, bounce_set)

        shots = []
        for shot_no, (s_start, s_end) in enumerate(shot_windows, start=1):
            is_serve = False
            if shot_no == 1:
                matrix = homography_matrices[s_start]
                if matrix is not None and ball_track[s_start][0] is not None:
                    court_point = _to_court_point(ball_track[s_start], matrix)
                    is_serve = _is_near_baseline(court_point, baseline_margin_cm)
            shots.append({
                'shot_no': shot_no,
                'start_frame': s_start,
                'end_frame': s_end,
                'is_serve': is_serve,
            })

        speeds = [s for i in range(r_start, r_end) if (s := ball_speed[i]) is not None]
        rallies.append({
            'rally_no': rally_no,
            'start_frame': r_start,
            'end_frame': r_end,
            'duration_s': (r_end - r_start) / fps,
            'num_shots': len(shots),
            'shots': shots,
            'avg_speed_kmh': (sum(speeds) / len(speeds)) if speeds else None,
            'max_speed_kmh': max(speeds) if speeds else None,
        })
    return rallies


def select_highlights(rallies, top_n=3):
    """
    Pick the most highlight-worthy rallies: the top_n with the fastest single
    shot, and the top_n with the longest duration. A rally can qualify under
    both criteria at once.
    :params
        rallies: list of rally dicts as returned by analyze_rallies
        top_n: how many rallies to pick per criterion
    :return
        list of dicts {rally, reasons}, sorted by start_frame (chronological,
        so clips can be shown/cut in the order they occur in the source video).
        `reasons` is a sorted list subset of ['fastest', 'longest'].
    """
    by_speed = sorted((r for r in rallies if r['max_speed_kmh'] is not None),
                       key=lambda r: r['max_speed_kmh'], reverse=True)[:top_n]
    by_duration = sorted(rallies, key=lambda r: r['duration_s'], reverse=True)[:top_n]

    picked = {}
    for r in by_speed:
        picked.setdefault(r['rally_no'], {'rally': r, 'reasons': set()})['reasons'].add('fastest')
    for r in by_duration:
        picked.setdefault(r['rally_no'], {'rally': r, 'reasons': set()})['reasons'].add('longest')

    return [{'rally': p['rally'], 'reasons': sorted(p['reasons'])}
            for p in sorted(picked.values(), key=lambda p: p['rally']['start_frame'])]
