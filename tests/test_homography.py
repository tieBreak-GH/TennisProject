import cv2
import pytest

from court_reference import CourtReference
from homography import get_trans_matrix, refer_kps


def test_get_trans_matrix_recovers_reference_points_when_image_matches_reference():
    # If the "detected" points are exactly the reference points, the fitted
    # homography should be close to identity and round-trip every keypoint.
    ref = CourtReference()
    points = list(ref.key_points)

    matrix = get_trans_matrix(points)

    assert matrix is not None
    transformed = cv2.perspectiveTransform(refer_kps, matrix).squeeze(1)
    for i, expected in enumerate(ref.key_points):
        assert transformed[i] == pytest.approx(expected, abs=1.0)


def test_get_trans_matrix_returns_none_without_any_points():
    assert get_trans_matrix([None] * 14) is None


def test_get_trans_matrix_returns_none_with_only_one_configs_points():
    # A candidate config needs at least one *other* known point to sanity-check
    # its fitted matrix against - `dists` stays empty and the candidate is
    # skipped (`if not dists: continue`) if the 4 points it was fit from are
    # the only ones available, even though the fit itself would succeed.
    ref = CourtReference()
    points = [None] * 14
    for idx in (8, 9, 10, 11):  # court_conf[5]'s four point indices
        points[idx] = ref.key_points[idx]

    assert get_trans_matrix(points) is None


def test_get_trans_matrix_succeeds_with_one_extra_verification_point():
    ref = CourtReference()
    points = [None] * 14
    for idx in (8, 9, 10, 11, 5):  # config 5's points, plus one extra to verify against
        points[idx] = ref.key_points[idx]

    assert get_trans_matrix(points) is not None
