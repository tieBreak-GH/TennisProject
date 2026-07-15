import cv2
import numpy as np

import config
from court_reference import CourtReference

_court_ref = CourtReference()
_BASELINE_TOP_Y = _court_ref.baseline_top[0][1]
_BASELINE_BOTTOM_Y = _court_ref.baseline_bottom[0][1]
_NET_Y = _court_ref.net[0][1]
_TOP_SERVICE_Y = _court_ref.top_inner_line[0][1]
_BOTTOM_SERVICE_Y = _court_ref.bottom_inner_line[0][1]
_CENTER_X = _court_ref.middle_line[0][0]
_LEFT_SINGLES_X = _court_ref.left_inner_line[0][0]
_RIGHT_SINGLES_X = _court_ref.right_inner_line[0][0]


def _to_court_point(point, matrix):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt_trans = cv2.perspectiveTransform(pt, matrix)
    return pt_trans[0, 0]


def _is_near_baseline(court_point, margin_cm=config.SERVE_BASELINE_MARGIN_CM):
    """
    Whether a court-plane point (CourtReference units, ~cm) falls within
    margin_cm of either baseline - used to sanity-check a serve candidate's
    starting position.
    """
    y = court_point[1]
    return abs(y - _BASELINE_TOP_Y) <= margin_cm or abs(y - _BASELINE_BOTTOM_Y) <= margin_cm


def _target_service_box(serve_x, serve_baseline):
    """
    A serve must land in the service box diagonally opposite the server:
    same half (left/right of the center line) mirrored across, on the
    receiving baseline's service line. Verified via the court's 180-degree
    rotational symmetry: a server at the bottom baseline right of center
    (deuce) targets the top baseline's LEFT box, and vice versa.
    :params
        serve_x: server's court-plane x position (approximated by the ball's
            position at the start of the serve)
        serve_baseline: 'top' or 'bottom' - which baseline the serve was hit from
    :return
        (x_min, x_max, y_min, y_max) of the target service box, in
        CourtReference court-plane units (~cm)
    """
    x_min, x_max = (_LEFT_SINGLES_X, _CENTER_X) if serve_x > _CENTER_X else (_CENTER_X, _RIGHT_SINGLES_X)
    y_min, y_max = (_NET_Y, _BOTTOM_SERVICE_Y) if serve_baseline == 'top' else (_TOP_SERVICE_Y, _NET_Y)
    return x_min, x_max, y_min, y_max


def _classify_point_in_box(point, box, margin_cm):
    """
    Classify a court-plane point against a box, with a "too close to call"
    margin around the edges (homography + bounce-frame detection error
    means a hard in/out call right at the line isn't trustworthy).
    :return
        'in' if comfortably inside, 'out' if comfortably outside, else
        'belirsiz' (uncertain) when within margin_cm of any edge
    """
    x, y = point
    x_min, x_max, y_min, y_max = box
    inside_margin = min(x - x_min, x_max - x, y - y_min, y_max - y)
    if inside_margin > margin_cm:
        return 'in'
    if inside_margin < -margin_cm:
        return 'out'
    return 'belirsiz'


def _call_serve_line(serve_court_point, bounce_frame, ball_track, bounce_set, homography_matrices, margin_cm):
    """
    Call a serve in/out against its target service box (see
    _target_service_box), or None if there's no reliable bounce to check
    (segment_shots sets a shot's end_frame to the bounce frame that closed
    it - if the rally never registered a bounce, end_frame is just the
    rally's end, not a real bounce, so a call can't be made).
    :params
        serve_court_point: the serve's starting position, already projected
            to court-plane coords (see analyze_rallies)
        bounce_frame: candidate bounce frame index (a shot's end_frame)
        bounce_set: set of frame indices where the ball actually bounced
    :return
        'in', 'out', 'belirsiz', or None
    """
    if bounce_frame not in bounce_set or ball_track[bounce_frame][0] is None:
        return None
    matrix = homography_matrices[bounce_frame]
    if matrix is None:
        return None
    bounce_point = _to_court_point(ball_track[bounce_frame], matrix)
    serve_baseline = 'top' if abs(serve_court_point[1] - _BASELINE_TOP_Y) < abs(serve_court_point[1] - _BASELINE_BOTTOM_Y) else 'bottom'
    box = _target_service_box(serve_court_point[0], serve_baseline)
    return _classify_point_in_box(bounce_point, box, margin_cm)


def segment_rallies(ball_track, fps, max_gap_seconds=config.RALLY_MAX_GAP_SECONDS):
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
                     max_gap_seconds=config.RALLY_MAX_GAP_SECONDS, baseline_margin_cm=config.SERVE_BASELINE_MARGIN_CM,
                     line_call_margin_cm=config.LINE_CALL_MARGIN_CM):
    """
    Segment the video into rallies and shots, and label each rally's first
    shot as a serve if its starting ball position projects near a baseline.
    This is the main defense against clips that start mid-rally: the first
    shot of the video trivially looks like "after a gap", but if it doesn't
    start near a baseline it is not labeled a serve. Serves are additionally
    given an estimated in/out line call against their target service box
    (see _call_serve_line) - deliberately scoped to serves only, since
    bounce-frame precision and homography error make a general in/out call
    on every shot unreliable.
    :params
        ball_track: list of (x, y) ball pixel coordinates per frame
        bounces: iterable of frame indices where the ball bounces
        homography_matrices: list of image->court homography matrices per frame
        ball_speed: list of speed values (km/h) per frame, as returned by
            speed_estimator.get_ball_speed
        fps: video frame rate
        max_gap_seconds: see segment_rallies
        baseline_margin_cm: see _is_near_baseline
        line_call_margin_cm: see _classify_point_in_box
    :return
        list of dicts, one per rally:
        {rally_no, start_frame, end_frame, duration_s, num_shots,
         shots: [{shot_no, start_frame, end_frame, is_serve, line_call}],
         avg_speed_kmh, max_speed_kmh}
        line_call is 'in' / 'out' / 'belirsiz' for a called serve, or None
        (not a serve, or no reliable bounce to call it against).
    """
    rally_windows = segment_rallies(ball_track, fps, max_gap_seconds)
    bounce_set = set(bounces)

    rallies = []
    for rally_no, (r_start, r_end) in enumerate(rally_windows, start=1):
        shot_windows = segment_shots(r_start, r_end, bounce_set)

        shots = []
        for shot_no, (s_start, s_end) in enumerate(shot_windows, start=1):
            is_serve = False
            line_call = None
            if shot_no == 1:
                matrix = homography_matrices[s_start]
                if matrix is not None and ball_track[s_start][0] is not None:
                    court_point = _to_court_point(ball_track[s_start], matrix)
                    is_serve = _is_near_baseline(court_point, baseline_margin_cm)
                    if is_serve:
                        line_call = _call_serve_line(court_point, s_end, ball_track, bounce_set,
                                                      homography_matrices, line_call_margin_cm)
            shots.append({
                'shot_no': shot_no,
                'start_frame': s_start,
                'end_frame': s_end,
                'is_serve': is_serve,
                'line_call': line_call,
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
