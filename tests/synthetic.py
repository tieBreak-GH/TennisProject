"""
Synthetic camera + projectile-motion helpers, shared by the camera
calibration and 3D trajectory tests (docs/uygulama_plani.md Faz 0.2).

These build a *known* pinhole camera and a *known* parabolic ball path,
then project the path into pixel coordinates - the reverse of what
camera_calib.py / trajectory_3d.py will need to recover. Fitting code can
be checked against ground truth without any model weights or real video.

Conventions:
    World frame: court plane is z=0 (cm, matching CourtReference units),
        x = court width axis, y = baseline-to-baseline axis, z = height
        above the ground.
    Camera frame (OpenCV-style): x right, y down, z forward (out of the
        lens). X_cam = R @ (X_world - C) = R @ X_world + t, t = -R @ C.
    Pixel: K @ X_cam, then perspective-divide.
"""
import numpy as np

GRAVITY_CMS2 = 981.0  # cm/s^2, downward (-z)


def make_camera(height_cm, pitch_deg, focal_px, image_size, camera_xy_cm=(0.0, 0.0)):
    """
    Build a pinhole camera looking down-court, tilted toward the ground.
    :params
        height_cm: camera height above the court plane (world z), cm
        pitch_deg: downward tilt from horizontal (0 = looking flat along
            +y, 90 = looking straight down) - must stay well under 90 to
            keep the camera basis non-degenerate
        focal_px: focal length in pixels (square pixels, fx == fy)
        image_size: (width, height) in pixels; principal point is centered
        camera_xy_cm: (x, y) world position of the camera's ground
            projection, cm - typically behind a baseline, on/near the center line
    :return
        K (3x3), R (3x3, world->camera rotation), t (3,, = -R @ C),
        C (3,, camera center in world coords), H_court2img (3x3 homography
        mapping court-plane (x, y, 1) -> homogeneous pixel coords, same
        convention as court_detection_net.infer_frame's returned matrix)
    """
    width, height = image_size
    p = np.deg2rad(pitch_deg)
    cx, cy = camera_xy_cm
    C = np.array([cx, cy, height_cm], dtype=np.float64)

    forward = np.array([0.0, np.cos(p), -np.sin(p)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)  # rows = camera axes in world coords

    t = -R @ C
    K = np.array([[focal_px, 0.0, width / 2.0],
                  [0.0, focal_px, height / 2.0],
                  [0.0, 0.0, 1.0]])

    H_court2img = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return K, R, t, C, H_court2img


def project_points(points_3d, K, R, t):
    """
    Project world-frame 3D points (cm) to pixel coordinates.
    :params
        points_3d: (N, 3) array, world cm
        K, R, t: as returned by make_camera
    :return
        (N, 2) array of (u, v) pixel coordinates
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    cam = points_3d @ R.T + t
    uv_h = cam @ K.T
    return uv_h[:, :2] / uv_h[:, 2:3]


def make_parabola(p0_cm, v0_cms, fps, n_frames, g=GRAVITY_CMS2):
    """
    Generate a projectile-motion ball path: p(t) = p0 + v0*t + 0.5*a*t^2,
    a = (0, 0, -g).
    :params
        p0_cm: (3,) start position, world cm
        v0_cms: (3,) initial velocity, cm/s
        fps: frames per second
        n_frames: number of samples
        g: gravitational acceleration, cm/s^2 (positive scalar, applied as -g in z)
    :return
        (points, times): points is (n_frames, 3) world cm, times is (n_frames,) seconds from p0
    """
    times = np.arange(n_frames, dtype=np.float64) / fps
    p0 = np.asarray(p0_cm, dtype=np.float64)
    v0 = np.asarray(v0_cms, dtype=np.float64)
    a = np.array([0.0, 0.0, -g])
    points = p0 + np.outer(times, v0) + 0.5 * np.outer(times ** 2, a)
    return points, times
