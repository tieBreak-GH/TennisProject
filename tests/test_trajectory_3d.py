import numpy as np
import pytest

from camera_calib import estimate_focal, decompose_pose
from trajectory_3d import (
    fit_segment_trajectory, segment_speed_series, segment_peak_speed,
    is_reliable_fit, GRAVITY_CMS2,
)
from tests.synthetic import make_camera, make_parabola, project_points

IMAGE_SIZE = (1920, 1080)


def _make_segment(p0, v0, fps=60, n_frames=8, height_cm=250, pitch_deg=15,
                   focal_px=1000, camera_xy_cm=(0, -800)):
    """Build a synthetic flight segment: known camera + known parabola -> pixels."""
    K, R, t, C, H = make_camera(height_cm=height_cm, pitch_deg=pitch_deg, focal_px=focal_px,
                                 image_size=IMAGE_SIZE, camera_xy_cm=camera_xy_cm)
    points, times = make_parabola(p0, v0, fps=fps, n_frames=n_frames)
    uv = project_points(points, K, R, t)
    return dict(K=K, R=R, t=t, C=C, times=times, uv=uv)


def test_fit_recovers_ground_truth_velocity_noiseless():
    v0_true = np.array([300.0, 1800.0, 600.0])
    seg = _make_segment(p0=(0, 200, 100), v0=v0_true)

    fit = fit_segment_trajectory(seg['times'], seg['uv'], seg['K'], seg['R'], seg['t'], seg['C'])

    assert fit is not None
    assert fit['v0'] == pytest.approx(v0_true, abs=1e-4)
    assert fit['rmse_px'] < 1e-6
    assert is_reliable_fit(fit)


def test_fit_recovers_speed_within_tolerance():
    v0_true = np.array([200.0, 1500.0, 900.0])
    expected_speed_kmh = np.linalg.norm(v0_true) * 0.036
    seg = _make_segment(p0=(50, 100, 50), v0=v0_true)

    fit = fit_segment_trajectory(seg['times'], seg['uv'], seg['K'], seg['R'], seg['t'], seg['C'])
    peak = segment_peak_speed(fit)

    assert peak == pytest.approx(expected_speed_kmh, rel=1e-3)


def test_fit_degrades_gracefully_under_pixel_noise():
    # A segment long enough (500ms @ 60fps) for the parabola's curvature to
    # dominate over 1px detector noise - see trajectory_3d.fit_segment_trajectory's
    # docstring: short segments and near-radial motion both need much more
    # data to converge, which is exactly what speed_std_kmh/is_reliable_fit
    # are for (tested separately below).
    rng = np.random.default_rng(0)
    v0_true = np.array([1800.0, 700.0, 500.0])  # mostly cross-frame motion, not radial
    seg = _make_segment(p0=(0, 150, 80), v0=v0_true, n_frames=30)

    noisy_uv = seg['uv'] + rng.normal(scale=1.0, size=seg['uv'].shape)  # +-1px detector noise
    fit = fit_segment_trajectory(seg['times'], noisy_uv, seg['K'], seg['R'], seg['t'], seg['C'])

    assert fit is not None
    recovered_speed = np.linalg.norm(fit['v0']) * 0.036
    expected_speed = np.linalg.norm(v0_true) * 0.036
    assert recovered_speed == pytest.approx(expected_speed, rel=0.1)
    assert is_reliable_fit(fit)


def test_is_reliable_fit_rejects_confidently_wrong_radial_motion_fit():
    # The failure mode that motivated speed_std_kmh: a ball moving mostly
    # along the camera's viewing axis (radial motion) barely shifts in
    # pixels, so 1px noise creates huge along-ray speed error - while
    # rmse_px stays deceptively low, because the noise is absorbed into an
    # incorrect depth/velocity rather than showing up as reprojection
    # residual. is_reliable_fit must catch this via speed_std_kmh even
    # though rmse_px alone would wrongly call it a good fit.
    rng = np.random.default_rng(0)
    v0_radial = np.array([250.0, 1700.0, 700.0])  # dominant component along the camera's forward axis
    seg = _make_segment(p0=(0, 150, 80), v0=v0_radial, n_frames=12)

    noisy_uv = seg['uv'] + rng.normal(scale=1.0, size=seg['uv'].shape)
    fit = fit_segment_trajectory(seg['times'], noisy_uv, seg['K'], seg['R'], seg['t'], seg['C'])

    assert fit is not None
    recovered_speed = np.linalg.norm(fit['v0']) * 0.036
    expected_speed = np.linalg.norm(v0_radial) * 0.036
    assert abs(recovered_speed - expected_speed) / expected_speed > 0.3  # confirms it's actually badly wrong
    assert not is_reliable_fit(fit)  # ... and is_reliable_fit correctly flags it despite low rmse_px


def test_fit_returns_none_with_fewer_than_three_samples():
    seg = _make_segment(p0=(0, 200, 100), v0=(300, 1800, 600), n_frames=2)
    fit = fit_segment_trajectory(seg['times'], seg['uv'], seg['K'], seg['R'], seg['t'], seg['C'])
    assert fit is None


def test_is_reliable_fit_flags_low_frame_count_high_rmse_and_high_speed_uncertainty():
    good_fit = {'n': 8, 'rmse_px': 1.0, 'v0': np.array([1000.0, 0, 0]), 'speed_std_kmh': 1.0}
    assert is_reliable_fit(good_fit)  # speed=36km/h, std=1km/h -> ratio ~3%, well under 15%

    too_few_frames = {**good_fit, 'n': 3}
    assert not is_reliable_fit(too_few_frames)

    bad_reprojection = {**good_fit, 'rmse_px': 20.0}
    assert not is_reliable_fit(bad_reprojection)

    high_speed_uncertainty = {**good_fit, 'speed_std_kmh': 20.0}  # ratio ~55%, way over 15%
    assert not is_reliable_fit(high_speed_uncertainty)

    assert not is_reliable_fit(None)


def test_is_reliable_fit_rejects_physically_impossible_speed_despite_tiny_uncertainty():
    # Discovered on a real broadcast match video (Faz 5.1): a long-lens,
    # behind-the-baseline camera makes nearly every rally shot near-radial,
    # which is ill-conditioned enough that scipy's LM solver can converge to
    # a wildly wrong depth/velocity (implied ball positions kilometers from
    # the court) that still reprojects well AND has a deceptively tiny local
    # speed_std_kmh/speed ratio - the covariance estimate only reflects local
    # curvature around whatever optimum was found, not the global ambiguity.
    # No real tennis shot exceeds ~270 km/h, so a fit claiming 930 km/h with
    # 0.04% relative uncertainty must still be rejected.
    impossible_fit = {'n': 28, 'rmse_px': 2.2, 'v0': np.array([0.0, 0.0, 25833.6]),
                       'speed_std_kmh': 0.37}
    assert not is_reliable_fit(impossible_fit)


def test_segment_speed_series_matches_analytic_projectile_speed():
    v0 = np.array([0.0, 0.0, 500.0])  # straight up
    times = np.array([0.0, 0.5, 1.0])
    speeds = segment_speed_series(v0, times)

    expected = np.array([np.linalg.norm(v0 + np.array([0, 0, -GRAVITY_CMS2]) * t) * 0.036 for t in times])
    assert speeds == pytest.approx(expected)


# --- The "killer test" (docs/uygulama_plani.md Faz 3.5 / 5.2): the whole
# point of the 3D approach is that recovered speed must NOT depend on how
# high the camera was mounted, unlike the existing 2D ground-projection
# method (speed_estimator.get_ball_speed), which the architecture report
# shows overstates speed by an amount that grows as camera height shrinks.
# This runs the full recoverable path: estimate_focal + decompose_pose from
# nothing but the homography (no ground-truth K/pose), then fits the
# trajectory - exactly what Faz 4 will wire into the real pipeline.
@pytest.mark.parametrize("height_cm,pitch_deg,camera_xy_cm", [
    (200.0, 8.0, (0, -700)),      # low tripod - worst case for the 2D method
    (500.0, 18.0, (0, -800)),
    (1200.0, 30.0, (0, -1000)),   # high broadcast camera - best case for the 2D method
])
def test_3d_speed_is_invariant_to_camera_height(height_cm, pitch_deg, camera_xy_cm):
    v0_true = np.array([400.0, 2200.0, 500.0])  # a single, fixed real-world shot
    expected_speed_kmh = np.linalg.norm(v0_true) * 0.036

    K_true, R_true, t_true, C_true, H = make_camera(
        height_cm=height_cm, pitch_deg=pitch_deg, focal_px=1000,
        image_size=IMAGE_SIZE, camera_xy_cm=camera_xy_cm)
    points, times = make_parabola((0, 200, 100), v0_true, fps=60, n_frames=8)
    uv = project_points(points, K_true, R_true, t_true)

    focal_est = estimate_focal(H, IMAGE_SIZE)
    K_est = np.array([[focal_est, 0, IMAGE_SIZE[0] / 2],
                      [0, focal_est, IMAGE_SIZE[1] / 2],
                      [0, 0, 1]])
    R, t, C = decompose_pose(H, K_est)

    fit = fit_segment_trajectory(times, uv, K_est, R, t, C)

    assert fit is not None
    recovered_speed = np.linalg.norm(fit['v0']) * 0.036
    # Same tolerance regardless of camera height - this is the point: a 4x
    # height difference (200cm vs 1200cm) produces no meaningful speed drift.
    assert recovered_speed == pytest.approx(expected_speed_kmh, rel=1e-2)
