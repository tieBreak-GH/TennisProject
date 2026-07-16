import numpy as np
import pytest

from camera_calib import estimate_focal, decompose_pose
from tests.synthetic import make_camera

IMAGE_SIZE = (1920, 1080)

# A range of realistic broadcast/handheld camera setups (height, pitch, focal).
# tests/synthetic.make_camera never rolls the camera (its right vector is
# always exactly the world x-axis), which makes the h1.omega.h2=0 constraint
# in estimate_focal structurally degenerate (h1[2] == 0 exactly) for every
# case here - that's a property of this synthetic camera model, not a bug;
# the h1.omega.h1 == h2.omega.h2 constraint alone carries the signal, and
# estimate_focal's denom-near-zero guard already falls back to it.
CAMERA_CASES = [
    dict(height_cm=700, pitch_deg=22, focal_px=950, camera_xy_cm=(0, -900)),
    dict(height_cm=1500, pitch_deg=35, focal_px=1200, camera_xy_cm=(50, -1000)),
    dict(height_cm=300, pitch_deg=12, focal_px=800, camera_xy_cm=(0, -700)),
    dict(height_cm=2500, pitch_deg=45, focal_px=1500, camera_xy_cm=(-100, -1500)),
]


@pytest.mark.parametrize("case", CAMERA_CASES)
def test_estimate_focal_recovers_ground_truth(case):
    K, R, t, C, H = make_camera(image_size=IMAGE_SIZE, **case)
    focal = estimate_focal(H, IMAGE_SIZE)
    assert focal == pytest.approx(case['focal_px'], rel=1e-3)


@pytest.mark.parametrize("case", CAMERA_CASES)
def test_decompose_pose_recovers_camera_height_with_ground_truth_k(case):
    K, R_true, t_true, C_true, H = make_camera(image_size=IMAGE_SIZE, **case)
    R, t, C = decompose_pose(H, K)

    assert C is not None
    assert C[2] == pytest.approx(case['height_cm'], rel=1e-3)
    assert C == pytest.approx(C_true, abs=1e-3)
    assert R == pytest.approx(R_true, abs=1e-6)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("case", CAMERA_CASES)
def test_full_pipeline_estimated_focal_still_recovers_height(case):
    # End-to-end: estimate K from the homography alone (no ground truth),
    # then decompose pose with that estimated K - the actually-usable path.
    K_true, R_true, t_true, C_true, H = make_camera(image_size=IMAGE_SIZE, **case)

    focal_est = estimate_focal(H, IMAGE_SIZE)
    K_est = np.array([[focal_est, 0, IMAGE_SIZE[0] / 2],
                       [0, focal_est, IMAGE_SIZE[1] / 2],
                       [0, 0, 1]])
    R, t, C = decompose_pose(H, K_est)

    assert C is not None
    assert C[2] == pytest.approx(case['height_cm'], rel=1e-2)


def test_estimate_focal_returns_none_for_fronto_parallel_homography():
    # A pure similarity transform (no perspective component) is the classic
    # degenerate case: a single homography of a fronto-parallel plane can't
    # recover focal length, since scale and depth are inseparable.
    fronto_parallel_H = np.array([[500.0, 0.0, 960.0],
                                   [0.0, 500.0, 540.0],
                                   [0.0, 0.0, 1.0]])
    assert estimate_focal(fronto_parallel_H, IMAGE_SIZE) is None


def test_decompose_pose_returns_none_for_degenerate_homography():
    degenerate_H = np.zeros((3, 3))
    degenerate_H[2, 2] = 1.0
    K = np.array([[900.0, 0, 960], [0, 900.0, 540], [0, 0, 1]])
    R, t, C = decompose_pose(degenerate_H, K)
    assert (R, t, C) == (None, None, None)
