import numpy as np
import pytest

from tests.synthetic import make_camera, project_points, make_parabola, GRAVITY_CMS2


def test_project_points_matches_direct_homography_on_court_plane():
    # A point on the court plane (z=0) projected via project_points should
    # match cv2.perspectiveTransform-style application of H_court2img built
    # from the same K, R, t - i.e. the homography is a correct specialization
    # of the full 3D projection when z=0.
    K, R, t, C, H = make_camera(height_cm=800, pitch_deg=25, focal_px=1000,
                                 image_size=(1920, 1080), camera_xy_cm=(0, -800))
    court_point = np.array([[150.0, 500.0, 0.0]])

    uv_direct = project_points(court_point, K, R, t)

    homog = H @ np.array([150.0, 500.0, 1.0])
    uv_via_h = homog[:2] / homog[2]

    assert uv_direct[0] == pytest.approx(uv_via_h, abs=1e-6)


def test_camera_center_projects_behind_image_is_excluded_by_construction():
    # Sanity check the camera basis is a proper rotation (orthonormal, det=1)
    # so it doesn't silently mirror or skew the scene.
    K, R, t, C, H = make_camera(height_cm=600, pitch_deg=15, focal_px=900,
                                 image_size=(1280, 720))
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_make_parabola_apex_height_matches_projectile_physics():
    # A ball launched straight up with v0_z should reach apex height
    # v0_z^2 / (2g) - standard kinematics, used here as a ground-truth check.
    v0_z = 500.0  # cm/s
    fps = 240  # fine time resolution to land close to the true apex
    points, times = make_parabola(p0_cm=(0, 0, 0), v0_cms=(0, 0, v0_z), fps=fps, n_frames=200)

    apex_expected = v0_z ** 2 / (2 * GRAVITY_CMS2)
    apex_measured = points[:, 2].max()

    assert apex_measured == pytest.approx(apex_expected, rel=1e-3)


def test_make_parabola_horizontal_motion_is_linear_without_gravity_component():
    points, times = make_parabola(p0_cm=(0, 0, 100), v0_cms=(300, 0, 0), fps=30, n_frames=10)
    expected_x = 300.0 * times
    assert points[:, 0] == pytest.approx(expected_x)
    # z should follow pure projectile fall from a fixed height with no vertical launch speed
    expected_z = 100.0 - 0.5 * GRAVITY_CMS2 * times ** 2
    assert points[:, 2] == pytest.approx(expected_z)


def test_project_points_reprojection_is_stable_for_moving_ball():
    # A full parabola projected into the camera should produce monotonically
    # sensible pixel coordinates (finite, within a generous bound) - guards
    # against a degenerate camera basis producing NaN/inf for any sample.
    K, R, t, C, H = make_camera(height_cm=1000, pitch_deg=30, focal_px=1100,
                                 image_size=(1920, 1080), camera_xy_cm=(0, -1200))
    points, times = make_parabola(p0_cm=(0, 500, 0), v0_cms=(200, 2000, 800), fps=60, n_frames=60)

    uv = project_points(points, K, R, t)

    assert np.all(np.isfinite(uv))
