"""
Centralized tunable thresholds and post-processing parameters.

Previously scattered as inline defaults across ball_detector.py,
court_detection_net.py, person_detector.py, bounce_detector.py,
speed_estimator.py, rally_analyzer.py and postprocess.py - collected here so
they can be reviewed/tuned in one place. This is a pure refactor: every
value below is unchanged from what the code already used, so behavior is
identical (see the pytest suite in tests/, all still green after this move).

Deliberately NOT centralized: constants that are part of a pretrained
model's expected input/feature shape rather than a runtime threshold -
changing them without retraining would silently break that model.
That includes ball_detector.BallDetector's 640x360 model input size and
bounce_detector.BounceDetector.prepare_features's lag window (num=3).
"""

# -- Person detection (person_detector.py) --
# YOLO's confidence calibration runs much lower than the old Faster R-CNN's
# for small/distant players; the court-mask filter does the real work of
# rejecting non-players, so this threshold can stay low. See the note in
# person_detector.PersonDetector.detect_top_and_bottom_players.
PERSON_MIN_SCORE = 0.3

# -- Ball speed estimation (speed_estimator.py) --
BALL_SPEED_WINDOW_FRAMES = 5    # max frame gap used as the speed baseline
BALL_SPEED_MAX_KMH = 300        # outlier guard: speeds above this are dropped
BALL_SPEED_SMOOTH_WINDOW = 5    # rolling median window to reduce jitter

# -- Ball tracking (ball_detector.py) --
BALL_MAX_JUMP_PX = 80  # postprocess: max distance from the previous ball
                        # detection to accept a new candidate as the same ball

# -- Bounce detection (bounce_detector.py) --
BOUNCE_SCORE_THRESHOLD = 0.45
BOUNCE_EXTRAPOLATION_MAX_JUMP_PX = 80  # smooth_predictions: max jump allowed
                                        # right after extrapolating a gap

# -- Rally/serve segmentation (rally_analyzer.py) --
RALLY_MAX_GAP_SECONDS = 1.5     # ball-tracking gap that starts a new rally
SERVE_BASELINE_MARGIN_CM = 300  # how close a shot's start must be to a
                                 # baseline (court-plane cm) to count as a serve

# -- Court keypoint detection (court_detection_net.py) --
COURT_MAX_PROBE_FRAMES = 5       # frames tried per scene before giving up
COURT_HEATMAP_THRESHOLD = 170    # binarize the per-keypoint heatmap
COURT_HOUGH_MIN_DIST = 20
COURT_HOUGH_PARAM1 = 50
COURT_HOUGH_PARAM2 = 2
COURT_HOUGH_MIN_RADIUS = 10
COURT_HOUGH_MAX_RADIUS = 25

# -- Ball-position heatmap circle detection (ball_detector.py) --
BALL_HEATMAP_THRESHOLD = 127
BALL_HOUGH_MIN_DIST = 1
BALL_HOUGH_PARAM1 = 50
BALL_HOUGH_PARAM2 = 2
BALL_HOUGH_MIN_RADIUS = 2
BALL_HOUGH_MAX_RADIUS = 7

# -- Court line refinement (postprocess.py) --
REFINE_KPS_CROP_SIZE = 40
LINE_DETECT_BINARY_THRESHOLD = 155
LINE_HOUGH_THRESHOLD = 30
LINE_HOUGH_MIN_LENGTH = 10
LINE_HOUGH_MAX_GAP = 30
LINE_MERGE_MAX_DIST = 20
