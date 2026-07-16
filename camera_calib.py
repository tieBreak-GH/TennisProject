"""
Recover camera intrinsics (focal length) and extrinsics (pose, height)
from a single court homography - no calibration target or user input
needed, since CourtReference already gives real-world (cm) court
coordinates. This is what lets trajectory_3d.py turn each ball pixel into
a metric 3D ray instead of a ground-plane-only projection, which is the
root cause of the camera-height-dependent speed error described in
docs/mimari_fizik_gpu_degerlendirme.md §3. See docs/uygulama_plani.md
Faz 2 for the derivation and role in the pipeline.

Standard single-homography calibration (Zhang 2000, specialized to one
plane), assuming square pixels, zero skew, and principal point at the
image center - reasonable for un-distorted consumer/broadcast footage,
and doesn't require anything beyond the court homography the pipeline
already computes.

Convention: all functions here take H_c2i, the *court-plane (cm) ->
image (pixel)* homography - the inverse of what main.py's
homography_matrices stores (which is image -> court, see
speed_estimator._transform_point). Callers integrating this with the
pipeline must invert before calling (Faz 4).
"""
import numpy as np


def estimate_focal(H_c2i, image_size):
    """
    Estimate focal length (pixels) from a court-plane homography, using
    the two orthonormality constraints on its rotation columns:
        h1 . omega . h2 = 0
        h1 . omega . h1 = h2 . omega . h2
    specialized to omega = diag(1/f^2, 1/f^2, 1) under the square-pixel /
    zero-skew / centered-principal-point assumption (h1, h2 are H_c2i's
    first two columns, after shifting the principal point to the origin).
    :params
        H_c2i: 3x3 homography, court-plane (cm, x, y) -> homogeneous image pixel
        image_size: (width, height) in pixels, used to center the principal point
    :return
        focal length in pixels (average of whichever constraint(s) give a
        finite, positive estimate), or None if both are degenerate (e.g. a
        near fronto-parallel view, where a single planar homography carries
        no perspective information to solve for focal length)
    """
    width, height = image_size
    T = np.array([[1.0, 0.0, -width / 2.0],
                  [0.0, 1.0, -height / 2.0],
                  [0.0, 0.0, 1.0]])
    h1, h2 = (T @ H_c2i)[:, 0], (T @ H_c2i)[:, 1]

    f_sq_candidates = []
    denom1 = h1[2] * h2[2]
    if abs(denom1) > 1e-9:
        f_sq_candidates.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / denom1)

    denom2 = h2[2] ** 2 - h1[2] ** 2
    if abs(denom2) > 1e-9:
        f_sq_candidates.append(((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / denom2)

    positive = [f for f in f_sq_candidates if f > 0]
    if not positive:
        return None
    return float(np.mean(np.sqrt(positive)))


def _nearest_rotation(m):
    """Closest proper rotation matrix to m (orthogonal Procrustes), det=+1."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def decompose_pose(H_c2i, K):
    """
    Decompose a court homography into camera pose, given known intrinsics
    (see estimate_focal).
    :params
        H_c2i: 3x3 homography, court-plane (cm) -> homogeneous image pixel
        K: 3x3 intrinsics matrix
    :return
        (R, t, C): R is the world->camera rotation, t = -R @ C, and C is
        the camera center in court-plane (cm) world coordinates - C[2] is
        the camera height above the court in cm (the quantity that drives
        the speed error in the 2D ground-plane projection). Returns
        (None, None, None) if the homography is too degenerate (near-zero
        columns) or if neither sign convention places the camera above the
        court (physically required for broadcast/handheld tennis footage).
    """
    B = np.linalg.inv(K) @ H_c2i
    norm1, norm2 = np.linalg.norm(B[:, 0]), np.linalg.norm(B[:, 1])
    if norm1 < 1e-9 or norm2 < 1e-9:
        return None, None, None
    # H is only defined up to an overall scale, so this reciprocal-norm
    # lambda could have either sign - resolved below by requiring the
    # camera to be physically above the court (C[2] > 0).
    lam = 2.0 / (norm1 + norm2)

    for sign in (lam, -lam):
        r1, r2 = sign * B[:, 0], sign * B[:, 1]
        r3 = np.cross(r1, r2)
        R = _nearest_rotation(np.column_stack([r1, r2, r3]))
        t = sign * B[:, 2]
        C = -R.T @ t
        if C[2] > 0:
            return R, t, C
    return None, None, None
