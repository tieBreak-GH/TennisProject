"""
Recover a ball's true 3D flight path (and therefore its camera-height-
independent speed) within one bounce-to-bounce segment, from 2D pixel
detections plus the camera pose from camera_calib.py.

Root problem this solves (docs/mimari_fizik_gpu_degerlendirme.md §3):
speed_estimator.py projects the ball's pixel onto the *ground* plane via
the court homography, which is only valid for points actually on the
ground (z=0) - the ball is airborne (z=h>0), so that projection is
systematically wrong, and by an amount that depends on the camera's
height. Here, instead, each ball pixel is treated as a 3D ray from the
camera center; a projectile-motion model (constant gravity, no other
forces) is fit to a whole flight segment so its position is *actually*
airborne along that ray, not flattened onto the ground. The result is a
real 3D trajectory and velocity - independent of how high the camera was
mounted. See docs/uygulama_plani.md Faz 3.
"""
import numpy as np
from scipy.optimize import least_squares

GRAVITY_CMS2 = 981.0  # cm/s^2, matches tests/synthetic.py's convention
_A_GRAV = np.array([0.0, 0.0, -GRAVITY_CMS2])


def pixel_ray_dir(uv, K, R):
    """
    Direction (world frame, unit length) of the 3D ray from the camera
    center through pixel uv.
    :params
        uv: (u, v) pixel coordinates
        K: 3x3 intrinsics
        R: 3x3 world->camera rotation (see camera_calib.decompose_pose)
    :return
        (3,) unit vector in world coordinates; the ball's 3D position for
        this pixel is C + s*d for some depth s > 0 (camera_calib's C)
    """
    d_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d_world = R.T @ d_cam
    return d_world / np.linalg.norm(d_world)


def _project(points_3d, K, R, t):
    cam = points_3d @ R.T + t
    uv_h = cam @ K.T
    return uv_h[:, :2] / uv_h[:, 2:3]


def _linear_fit(times, ray_dirs, C):
    """
    Initial (exact, non-iterative) solve for p0, v0 and per-sample depths
    s_k, from the linear system obtained by equating the projectile model
    to the camera-ray parametrization:
        p0 + v0*t_k - s_k*d_k = C - 0.5*a*t_k^2
    3 equations per sample, 6+N unknowns - solvable for N >= 3.
    :return
        (p0, v0) or None if the system is under-determined (N < 3)
    """
    n = len(times)
    if n < 3:
        return None

    A = np.zeros((3 * n, 6 + n))
    b = np.zeros(3 * n)
    for k in range(n):
        rows = slice(3 * k, 3 * k + 3)
        A[rows, 0:3] = np.eye(3)
        A[rows, 3:6] = times[k] * np.eye(3)
        A[rows, 6 + k] = -ray_dirs[k]
        b[rows] = C - 0.5 * _A_GRAV * times[k] ** 2

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], x[3:6]


def fit_segment_trajectory(times, pixel_coords, K, R, t, C):
    """
    Fit a projectile-motion trajectory to one bounce-to-bounce (or
    vuruş-to-bounce) flight segment's ball detections.

    Reprojection error alone (rmse_px) is NOT a sufficient reliability
    signal here: when the ball's motion is nearly radial (close to the
    camera's viewing axis - e.g. a shot hit straight down-court from a
    baseline camera), pixel noise gets absorbed into an incorrect
    along-ray depth/speed rather than showing up as reprojection residual,
    so a bad fit can still report a deceptively low rmse_px (verified
    empirically: 1px noise on a radial-motion segment gave rmse_px=0.7 but
    a 64% speed error). This also fits worse on short segments regardless
    of motion direction - the parabola's curvature (the only cue that
    breaks the depth/speed ambiguity) needs enough time to become visible
    above the pixel noise floor (empirically: ~130ms segments were
    unusable, >=500ms segments were accurate to a couple percent under
    1px noise). So this also computes speed_std_kmh, a linearized
    parameter-uncertainty estimate from the fit's Jacobian, which tracked
    actual error well across both failure modes in that same testing and
    is what is_reliable_fit gates on.
    :params
        times: (N,) seconds, relative to any fixed reference (only
            differences matter) - typically frame_index/fps
        pixel_coords: (N, 2) observed ball pixel positions for this segment
        K, R, t, C: camera intrinsics/pose for this segment's scene (see
            camera_calib.py) - assumed constant within the segment, same
            assumption the pipeline already makes for the court homography
    :return
        dict {p0, v0, rmse_px, n, speed_std_kmh} - p0 (cm, world) and v0
        (cm/s, world) at times[0], rmse_px the reprojection RMSE, n the
        sample count, speed_std_kmh the estimated 1-sigma uncertainty on
        |v0|'s speed (km/h) - or None if there aren't enough samples
        (N < 3) to solve the system at all
    """
    times = np.asarray(times, dtype=np.float64)
    pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
    n = len(times)

    ray_dirs = np.array([pixel_ray_dir(uv, K, R) for uv in pixel_coords])
    linear = _linear_fit(times, ray_dirs, C)
    if linear is None:
        return None
    p0_lin, v0_lin = linear

    def residuals(params):
        p0, v0 = params[:3], params[3:6]
        points = p0 + np.outer(times, v0) + 0.5 * np.outer(times ** 2, _A_GRAV)
        uv_pred = _project(points, K, R, t)
        return (uv_pred - pixel_coords).ravel()

    sol = least_squares(residuals, np.concatenate([p0_lin, v0_lin]), method='lm')
    p0, v0 = sol.x[:3], sol.x[3:6]
    rmse_px = float(np.sqrt(np.mean(sol.fun ** 2)))

    # Linearized parameter uncertainty (standard nonlinear-least-squares
    # covariance approximation: residual_variance * pinv(J^T J)), then
    # propagated onto the scalar speed |v0| via its gradient v0/|v0|.
    dof = max(1, 2 * n - 6)
    residual_var = float(np.sum(sol.fun ** 2)) / dof
    cov = residual_var * np.linalg.pinv(sol.jac.T @ sol.jac)
    v0_cov = cov[3:6, 3:6]
    speed_cms = np.linalg.norm(v0)
    if speed_cms > 1e-9:
        grad = v0 / speed_cms
        speed_var_cms = max(float(grad @ v0_cov @ grad), 0.0)
    else:
        speed_var_cms = float(np.trace(v0_cov))
    speed_std_kmh = np.sqrt(speed_var_cms) * 0.036

    return {'p0': p0, 'v0': v0, 'rmse_px': rmse_px, 'n': n, 'speed_std_kmh': speed_std_kmh}


def segment_speed_series(v0, times, t0=0.0):
    """
    Instantaneous speed (km/h) at each time in `times`, from the fitted
    v0 (cm/s at t0) under constant-gravity projectile motion:
    v(t) = v0 + a*(t - t0).
    :params
        v0: (3,) cm/s, as returned by fit_segment_trajectory
        times: (N,) seconds (same reference as the fit's `times`)
        t0: the time at which v0 applies (fit_segment_trajectory's times[0]
            reference point) - pass the same value used there if not 0
    :return
        (N,) array of speeds in km/h
    """
    times = np.asarray(times, dtype=np.float64)
    velocity = v0 + np.outer(times - t0, _A_GRAV)
    return np.linalg.norm(velocity, axis=1) * 0.036  # cm/s -> km/h


def segment_peak_speed(fit_result, sample_times=None):
    """
    Single peak-speed number (km/h) for a fitted segment - the analogue of
    speed_estimator.get_shot_max_speed's per-segment HUD value, but derived
    from the true 3D velocity instead of a 2D ground projection.
    :params
        fit_result: dict from fit_segment_trajectory
        sample_times: times to evaluate speed at (defaults to just t=0,
            i.e. |v0| - the launch/bounce instant, usually where a shot's
            speed peaks); pass the segment's actual per-frame times to
            search the whole flight instead
    :return
        peak speed in km/h
    """
    times = np.array([0.0]) if sample_times is None else np.asarray(sample_times)
    return float(np.max(segment_speed_series(fit_result['v0'], times)))


def is_reliable_fit(fit_result, min_frames=5, max_rmse_px=8.0, max_speed_std_ratio=0.15,
                     max_speed_kmh=300.0):
    """
    Whether a fit is trustworthy enough to use instead of falling back to
    the 2D ground-projection speed estimate.

    The primary statistical gate is relative speed uncertainty
    (speed_std_kmh / speed) - see fit_segment_trajectory's docstring for why
    rmse_px and frame count alone can't catch a confidently-wrong fit
    (near-radial ball motion, or a too-short flight segment, both showed low
    rmse_px despite large actual speed error in synthetic testing; the
    covariance-based speed_std_kmh tracked actual error well in both cases).

    max_speed_kmh is a second, non-statistical gate added after validating
    against a real broadcast-camera match video (Faz 5.1): with a long-lens,
    behind-the-baseline camera, essentially every rally shot moves close to
    the camera's viewing axis, which is exactly the near-radial case where
    monocular depth/velocity separation is ill-conditioned - and for that
    real footage, EVERY bounce-to-bounce segment converged to a wildly wrong
    solution (implied ball positions kilometers from the court, speeds from
    300 to 3*10^8 km/h), several of which still had a deceptively small
    speed_std_kmh/speed ratio (the covariance estimate only reflects local
    curvature around whatever optimum scipy's LM solver landed in - it
    can't see that the global least-squares landscape had multiple very
    different, similarly-well-reprojecting solutions). No tennis shot ever
    recorded exceeds ~270 km/h, so any fit above this ceiling is certainly
    wrong regardless of what its own uncertainty estimate claims.
    :params
        fit_result: dict from fit_segment_trajectory, or None
        min_frames: below this the system is barely/not over-determined
            (see _linear_fit's 3N >= 6+N solvability requirement)
        max_rmse_px: reprojection error above this indicates a bad
            homography, mistracked ball, or a genuinely non-projectile
            path (e.g. a net clip) the model doesn't fit well
        max_speed_std_ratio: reject if the estimated 1-sigma speed
            uncertainty exceeds this fraction of the fitted speed itself
        max_speed_kmh: hard physical-plausibility ceiling - reject
            regardless of how confident the fit claims to be
    """
    if fit_result is None or fit_result['n'] < min_frames or fit_result['rmse_px'] > max_rmse_px:
        return False
    speed_kmh = float(np.linalg.norm(fit_result['v0'])) * 0.036
    if speed_kmh > max_speed_kmh:
        return False
    if speed_kmh <= 1e-9:
        return fit_result['speed_std_kmh'] <= 1e-6
    return (fit_result['speed_std_kmh'] / speed_kmh) <= max_speed_std_ratio
