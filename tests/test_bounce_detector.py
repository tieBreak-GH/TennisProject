import numpy as np

from bounce_detector import BounceDetector


def _detector():
    # postprocess only touches its ind_bounce/preds args, not self.model,
    # so an unloaded detector (no path_model) is fine here.
    return BounceDetector()


def test_postprocess_keeps_isolated_indices_untouched():
    det = _detector()
    result = det.postprocess(np.array([2, 10, 20]), preds=np.zeros(21))
    assert result == [2, 10, 20]


def test_postprocess_collapses_consecutive_run_to_highest_pred_index():
    det = _detector()
    preds = np.zeros(8)
    preds[5], preds[6], preds[7] = 0.5, 0.9, 0.6
    result = det.postprocess(np.array([5, 6, 7]), preds)
    assert result == [6]


def test_postprocess_mixed_runs_and_isolated_indices():
    det = _detector()
    preds = np.zeros(11)
    preds[5], preds[6] = 0.3, 0.8
    result = det.postprocess(np.array([1, 5, 6, 10]), preds)
    assert result == [1, 6, 10]
