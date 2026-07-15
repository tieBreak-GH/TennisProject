import queue
import threading

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    """
    video = open_video(path_video)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    if scene_list == []:
        scene_list = [(video.base_timecode, video.duration)]
    scenes = [[x[0].frame_num, x[1].frame_num]for x in scene_list]
    return scenes


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


