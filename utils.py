import queue
import threading

import cv2
import numpy as np

import config


def _mean_pixel_distance(a, b):
    """Mean absolute per-pixel difference between two same-shape 2D arrays."""
    return np.abs(a.astype(np.int32) - b.astype(np.int32)).mean()


class SceneCutDetector:
    """
    Online, per-frame reimplementation of PySceneDetect's ContentDetector
    default behavior (equal-weighted HSV hue/saturation/luma mean pixel
    distance between consecutive frames, no edge component) - runs inline as
    frames are decoded elsewhere instead of PySceneDetect's own separate
    full-video decode pass.

    Approximation vs PySceneDetect: a cut is confirmed as soon as the score
    clears the threshold and min_scene_len frames have passed since the last
    cut, rather than PySceneDetect's default flash-merge filter (which can
    delay confirming a cut by several frames to merge nearby ones). Good
    enough for resetting per-scene court homography probing; not intended as
    a numeric match.
    """

    def __init__(self, threshold=config.SCENE_CUT_THRESHOLD, min_scene_len=config.SCENE_CUT_MIN_LEN):
        self._threshold = threshold
        self._min_scene_len = min_scene_len
        self._last_hsv = None
        self._last_cut = 0

    def is_cut(self, frame_num, frame):
        hue, sat, lum = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        if self._last_hsv is None:
            self._last_hsv = (hue, sat, lum)
            return False

        prev_hue, prev_sat, prev_lum = self._last_hsv
        score = (_mean_pixel_distance(hue, prev_hue)
                 + _mean_pixel_distance(sat, prev_sat)
                 + _mean_pixel_distance(lum, prev_lum)) / 3
        self._last_hsv = (hue, sat, lum)

        if score >= self._threshold and frame_num - self._last_cut >= self._min_scene_len:
            self._last_cut = frame_num
            return True
        return False


class ThreadedFrameReader:
    """
    Decodes video frames on a background thread into a bounded queue, so
    decoding frame i+1 overlaps with the caller processing frame i instead
    of the two taking turns on one thread (plain cv2.VideoCapture.read()
    blocks the caller for the full decode of each frame, idling whatever
    model inference / rendering it's driving).
    """
    _SENTINEL = None

    def __init__(self, path, queue_size=8):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count_hint = max(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), 1)
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            self._queue.put(frame)
        self._queue.put(self._SENTINEL)

    def __iter__(self):
        while True:
            frame = self._queue.get()
            if frame is self._SENTINEL:
                return
            yield frame

    def close(self):
        """
        Stop the reader and release the capture. Safe to call after the
        stream has been fully consumed (the thread has already exited) or
        after breaking out early (drains the queue so a reader thread
        blocked on a full queue can observe the stop signal and exit,
        instead of leaking a thread that leaks its own decoded frames).
        """
        self._stop.set()
        while self._thread.is_alive():
            try:
                self._queue.get(timeout=0.1)
            except queue.Empty:
                pass
        self.cap.release()


class ThreadedFrameWriter:
    """
    Hands frames to a cv2.VideoWriter on a background thread via a bounded
    FIFO queue (single producer, single consumer - order-preserving), so
    encode + disk I/O overlaps with the caller producing the next frame
    instead of blocking it.
    """
    _SENTINEL = object()

    def __init__(self, writer, queue_size=8):
        self._writer = writer
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                return
            try:
                self._writer.write(item)
            except Exception as e:  # surfaced to the caller by write()/release()
                self._error = e

    def write(self, frame):
        if self._error is not None:
            raise self._error
        self._queue.put(frame)

    def release(self):
        self._queue.put(self._SENTINEL)
        self._thread.join()
        self._writer.release()
        if self._error is not None:
            raise self._error


