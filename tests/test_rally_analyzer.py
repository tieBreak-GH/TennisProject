import numpy as np

from rally_analyzer import analyze_rallies, segment_rallies, segment_shots, select_highlights

IDENTITY = np.eye(3, dtype=np.float32)


def test_segment_rallies_splits_on_gap_longer_than_threshold():
    # fps=10, max_gap_seconds=1.5 -> max_gap_frames=15
    ball_track = [(0, 0)] * 5 + [(None, None)] * 20 + [(0, 0)] * 5
    rallies = segment_rallies(ball_track, fps=10, max_gap_seconds=1.5)
    assert rallies == [(0, 5), (25, 30)]


def test_segment_rallies_short_gap_does_not_split():
    # a 10-frame gap is under the 15-frame threshold - stays one rally
    ball_track = [(0, 0)] * 5 + [(None, None)] * 10 + [(0, 0)] * 5
    rallies = segment_rallies(ball_track, fps=10, max_gap_seconds=1.5)
    assert rallies == [(0, 20)]


def test_segment_rallies_no_tracked_frames_returns_empty():
    assert segment_rallies([(None, None)] * 5, fps=10) == []


def test_segment_shots_splits_on_bounce_frames():
    assert segment_shots(0, 10, [3, 7]) == [(0, 3), (3, 7), (7, 10)]


def test_segment_shots_ignores_bounces_outside_window():
    assert segment_shots(5, 10, [1, 3, 12]) == [(5, 10)]


def test_segment_shots_no_bounces_returns_whole_window():
    assert segment_shots(0, 10, []) == [(0, 10)]


def test_analyze_rallies_labels_first_shot_as_serve_near_baseline():
    # identity homography -> image point == court point, so placing the ball
    # exactly on the top baseline's y should trigger the near-baseline check
    ball_track = [(500.0, 561.0)]
    rallies = analyze_rallies(ball_track, bounces=[], homography_matrices=[IDENTITY],
                               ball_speed=[None], fps=10)
    assert len(rallies) == 1
    # is_serve is a numpy bool (from a numpy-array comparison upstream), so
    # compare by value rather than `is True`/`is False`
    assert bool(rallies[0]['shots'][0]['is_serve']) is True


def test_analyze_rallies_does_not_label_serve_away_from_baseline():
    ball_track = [(500.0, 1748.0)]  # net height, far from both baselines
    rallies = analyze_rallies(ball_track, bounces=[], homography_matrices=[IDENTITY],
                               ball_speed=[None], fps=10)
    assert bool(rallies[0]['shots'][0]['is_serve']) is False


def test_analyze_rallies_serve_line_call_in():
    # serve from bottom baseline (y=2935), x=1000 is right of center (832)
    # -> target box is the top-left service box: x in [423,832], y in [1110,1748]
    ball_track = [(1000.0, 2935.0), (600.0, 1400.0)]  # bounce well inside the box
    rallies = analyze_rallies(ball_track, bounces=[1], homography_matrices=[IDENTITY] * 2,
                               ball_speed=[None, None], fps=10)
    assert rallies[0]['shots'][0]['line_call'] == 'in'


def test_analyze_rallies_serve_line_call_out():
    # same serve, but bounce lands on the wrong (right) side of the target box
    ball_track = [(1000.0, 2935.0), (1000.0, 1400.0)]
    rallies = analyze_rallies(ball_track, bounces=[1], homography_matrices=[IDENTITY] * 2,
                               ball_speed=[None, None], fps=10)
    assert rallies[0]['shots'][0]['line_call'] == 'out'


def test_analyze_rallies_serve_line_call_uncertain_near_line():
    # bounce only 7cm inside the center line (832) - within the default
    # 20cm margin, so too close to call confidently
    ball_track = [(1000.0, 2935.0), (825.0, 1400.0)]
    rallies = analyze_rallies(ball_track, bounces=[1], homography_matrices=[IDENTITY] * 2,
                               ball_speed=[None, None], fps=10)
    assert rallies[0]['shots'][0]['line_call'] == 'belirsiz'


def test_analyze_rallies_serve_line_call_none_without_a_bounce():
    # a rally with no detected bounce has no real bounce frame to call against
    ball_track = [(1000.0, 2935.0)]
    rallies = analyze_rallies(ball_track, bounces=[], homography_matrices=[IDENTITY],
                               ball_speed=[None], fps=10)
    assert bool(rallies[0]['shots'][0]['is_serve']) is True
    assert rallies[0]['shots'][0]['line_call'] is None


def test_select_highlights_picks_top_n_per_criterion():
    rallies = [
        {'rally_no': 1, 'start_frame': 0, 'end_frame': 10, 'duration_s': 1.0, 'max_speed_kmh': 80},
        {'rally_no': 2, 'start_frame': 10, 'end_frame': 60, 'duration_s': 5.0, 'max_speed_kmh': 150},
        {'rally_no': 3, 'start_frame': 60, 'end_frame': 200, 'duration_s': 14.0, 'max_speed_kmh': 90},
        {'rally_no': 4, 'start_frame': 200, 'end_frame': 205, 'duration_s': 0.5, 'max_speed_kmh': None},
    ]
    highlights = select_highlights(rallies, top_n=1)
    picked = {h['rally']['rally_no']: h['reasons'] for h in highlights}
    assert picked == {2: ['fastest'], 3: ['longest']}


def test_select_highlights_rally_can_qualify_under_both_criteria():
    rallies = [
        {'rally_no': 1, 'start_frame': 0, 'end_frame': 10, 'duration_s': 1.0, 'max_speed_kmh': 80},
        {'rally_no': 2, 'start_frame': 10, 'end_frame': 60, 'duration_s': 5.0, 'max_speed_kmh': 150},
    ]
    highlights = select_highlights(rallies, top_n=2)
    picked = {h['rally']['rally_no']: h['reasons'] for h in highlights}
    assert picked == {1: ['fastest', 'longest'], 2: ['fastest', 'longest']}


def test_select_highlights_ignores_rallies_with_no_speed_data_for_fastest():
    rallies = [
        {'rally_no': 1, 'start_frame': 0, 'end_frame': 10, 'duration_s': 1.0, 'max_speed_kmh': None},
    ]
    highlights = select_highlights(rallies, top_n=3)
    picked = {h['rally']['rally_no']: h['reasons'] for h in highlights}
    assert picked == {1: ['longest']}
