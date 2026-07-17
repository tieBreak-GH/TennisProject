import numpy as np
import pytest

from speed_estimator import get_ball_speed_3d, estimate_ball_speed
from tests.synthetic import make_camera, make_parabola, project_points

IMAGE_SIZE = (1920, 1080)
FPS = 60
_A_GRAV = np.array([0.0, 0.0, -981.0])


def _make_scene(v0, p0=(0, 200, 100), n_frames=30, height_cm=500, pitch_deg=18, camera_xy_cm=(0, -800)):
    """One scene (constant homography), one bounce-to-bounce flight -> the
    inputs speed_estimator's dispatcher expects: ball_track (pixels),
    homography_matrices (image->court, i.e. inv of synthetic's court->image),
    scenes, and the flight's own per-frame times (for checking expected speed).
    """
    K, R, t, C, H_c2i = make_camera(height_cm=height_cm, pitch_deg=pitch_deg, focal_px=1000,
                                     image_size=IMAGE_SIZE, camera_xy_cm=camera_xy_cm)
    points, times = make_parabola(p0, v0, fps=FPS, n_frames=n_frames)
    uv = project_points(points, K, R, t)
    ball_track = [(float(x), float(y)) for x, y in uv]
    homography_matrices = [np.linalg.inv(H_c2i)] * n_frames
    scenes = [(0, n_frames)]
    return ball_track, homography_matrices, scenes, times


def test_get_ball_speed_3d_recovers_instantaneous_speed_for_reliable_segment():
    v0 = np.array([400.0, 2200.0, 500.0])
    ball_track, homography_matrices, scenes, times = _make_scene(v0)

    speeds_3d, camera_height_cm = get_ball_speed_3d(
        ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE)

    assert all(h == pytest.approx(500.0, rel=1e-2) for h in camera_height_cm)
    assert all(s is not None for s in speeds_3d)

    expected = [float(np.linalg.norm(v0 + _A_GRAV * t)) * 0.036 for t in times]
    for speed, exp in zip(speeds_3d, expected):
        assert speed == pytest.approx(exp, rel=0.05)


def test_get_ball_speed_3d_returns_none_without_court_homography():
    ball_track, homography_matrices, scenes, _ = _make_scene(np.array([400.0, 2200.0, 500.0]))
    homography_matrices = [None] * len(ball_track)

    speeds_3d, camera_height_cm = get_ball_speed_3d(
        ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE)

    assert all(s is None for s in speeds_3d)
    assert all(h is None for h in camera_height_cm)


def test_estimate_ball_speed_dispatcher_uses_3d_when_reliable():
    v0 = np.array([400.0, 2200.0, 500.0])
    ball_track, homography_matrices, scenes, _ = _make_scene(v0)

    speeds, method_per_frame, camera_height_cm = estimate_ball_speed(
        ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE)

    assert method_per_frame == ['3d'] * len(ball_track)
    assert all(s is not None for s in speeds)
    assert all(h == pytest.approx(500.0, rel=1e-2) for h in camera_height_cm)


def test_estimate_ball_speed_falls_back_to_2d_for_unreliable_segment():
    # A too-short segment (6 frames / 100ms @ 60fps) with cross-frame (not
    # radial) motion, plus pixel noise: trajectory_3d's own tests show short
    # segments alone destabilize the 3D fit regardless of motion direction
    # (see tests/test_trajectory_3d.py and docs/uygulama_plani.md Faz 3.4),
    # so is_reliable_fit should reject it - while the 2D ground-projection
    # method (not radial here, so not catastrophically wrong either) still
    # produces a value, proving the dispatcher actually falls back instead of
    # leaving these frames None.
    rng = np.random.default_rng(0)
    v0 = np.array([1800.0, 700.0, 500.0])
    ball_track, homography_matrices, scenes, _ = _make_scene(v0, p0=(0, 150, 80), n_frames=6)
    noisy_track = [(x + rng.normal(scale=1.0), y + rng.normal(scale=1.0)) for x, y in ball_track]

    speeds, method_per_frame, camera_height_cm = estimate_ball_speed(
        noisy_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE)

    assert '3d' not in method_per_frame
    assert '2d' in method_per_frame
    assert any(s is not None for s in speeds)


def test_estimate_ball_speed_method_3d_has_no_2d_fallback():
    rng = np.random.default_rng(0)
    v0 = np.array([1800.0, 700.0, 500.0])
    ball_track, homography_matrices, scenes, _ = _make_scene(v0, p0=(0, 150, 80), n_frames=6)
    noisy_track = [(x + rng.normal(scale=1.0), y + rng.normal(scale=1.0)) for x, y in ball_track]

    speeds, method_per_frame, _ = estimate_ball_speed(
        noisy_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE, method='3d')

    assert '2d' not in method_per_frame
    assert all(s is None for s, m in zip(speeds, method_per_frame) if m is None)


def test_estimate_ball_speed_method_2d_ignores_3d_entirely():
    v0 = np.array([400.0, 2200.0, 500.0])
    ball_track, homography_matrices, scenes, _ = _make_scene(v0)

    speeds, method_per_frame, camera_height_cm = estimate_ball_speed(
        ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE, method='2d')

    assert '3d' not in method_per_frame
    assert all(h is None for h in camera_height_cm)


# --- Faz 5.2 "killer test", run through the actual dispatcher (Faz 4's
# estimate_ball_speed) instead of the isolated trajectory_3d module (see
# tests/test_trajectory_3d.py's own version of this test): the same shot,
# filmed from 3 different camera heights, must yield the same speed via the
# 3D-first dispatcher, while the legacy 2D method it replaces gives a
# different (and wrong) answer at each height - this is the concrete,
# regression-proof answer to the original "doğru top hızını farklı kamera
# yüksekliklerinde nasıl ölçeriz" question.
@pytest.mark.parametrize("height_cm,pitch_deg,camera_xy_cm", [
    (200.0, 8.0, (0, -700)),
    (500.0, 18.0, (0, -800)),
    (1200.0, 30.0, (0, -1000)),
])
def test_dispatcher_3d_speed_is_invariant_to_camera_height(height_cm, pitch_deg, camera_xy_cm):
    v0 = np.array([400.0, 2200.0, 500.0])
    ball_track, homography_matrices, scenes, times = _make_scene(
        v0, height_cm=height_cm, pitch_deg=pitch_deg, camera_xy_cm=camera_xy_cm)
    mid = len(ball_track) // 2
    expected_speed_kmh = float(np.linalg.norm(v0 + _A_GRAV * times[mid])) * 0.036

    speeds, method_per_frame, _ = estimate_ball_speed(
        ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE)

    assert method_per_frame[mid] == '3d'
    assert speeds[mid] == pytest.approx(expected_speed_kmh, rel=1e-2)


def test_legacy_2d_method_diverges_across_camera_heights_that_dispatcher_does_not():
    v0 = np.array([400.0, 2200.0, 500.0])
    heights = [(200.0, 8.0, (0, -700)), (500.0, 18.0, (0, -800)), (1200.0, 30.0, (0, -1000))]

    legacy_by_height = {}
    dispatched_by_height = {}
    for height_cm, pitch_deg, camera_xy_cm in heights:
        ball_track, homography_matrices, scenes, times = _make_scene(
            v0, height_cm=height_cm, pitch_deg=pitch_deg, camera_xy_cm=camera_xy_cm)
        mid = len(ball_track) // 2
        expected_speed_kmh = float(np.linalg.norm(v0 + _A_GRAV * times[mid])) * 0.036

        legacy_speeds, _, _ = estimate_ball_speed(
            ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE, method='2d')
        dispatched_speeds, method_per_frame, _ = estimate_ball_speed(
            ball_track, homography_matrices, scenes, FPS, bounce_frames=(), image_size=IMAGE_SIZE)

        legacy_by_height[height_cm] = legacy_speeds[mid]
        dispatched_by_height[height_cm] = dispatched_speeds[mid]
        assert method_per_frame[mid] == '3d'
        assert dispatched_speeds[mid] == pytest.approx(expected_speed_kmh, rel=1e-2)

    # The old 2D method's speed for the *same physical shot* swings wildly
    # with camera height: at 200cm it's so wrong its own >300km/h outlier
    # guard discards it entirely; at 500cm vs 1200cm it differs by >50%.
    # This is the systematic, height-dependent error documented in
    # docs/mimari_fizik_gpu_degerlendirme.md Section 3.
    assert legacy_by_height[200.0] is None
    assert legacy_by_height[500.0] is not None and legacy_by_height[1200.0] is not None
    assert abs(legacy_by_height[500.0] - legacy_by_height[1200.0]) / legacy_by_height[1200.0] > 0.5

    # The dispatcher's 3D-first result for the same shot barely moves
    # (<3%, Faz 5's acceptance criterion) regardless of camera height.
    dispatched_values = list(dispatched_by_height.values())
    assert (max(dispatched_values) - min(dispatched_values)) / min(dispatched_values) < 0.03
