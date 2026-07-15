import cv2
from ultralytics import YOLO
from court_reference import CourtReference
from scipy import signal
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

import config

class PersonDetector():
    def __init__(self, device='cpu'):
        self.detection_model = YOLO('yolo11n.pt')
        self.device = device
        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0


    def detect_batch(self, images, person_min_score=config.PERSON_MIN_SCORE):
        """ Run YOLO on a list of images in one predict() call - batching the
        forward pass is what actually benefits from GPU parallelism; the
        per-frame court-mask filtering below stays a cheap CPU loop.
        :return
            list of (persons_boxes, probs), one entry per input image, in order
        """
        PERSON_CLASS = 0  # COCO "person" class
        results_list = self.detection_model.predict(images, classes=[PERSON_CLASS], conf=person_min_score,
                                                      device=self.device, verbose=False)
        return [([box.cpu().numpy() for box in results.boxes.xyxy],
                 [float(score) for score in results.boxes.conf])
                for results in results_list]

    def detect(self, image, person_min_score=config.PERSON_MIN_SCORE):
        return self.detect_batch([image], person_min_score)[0]

    def filter_top_bottom(self, image, inv_matrix, bboxes, probs, filter_players=False):
        matrix = cv2.invert(inv_matrix)[1]
        mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, image.shape[1::-1])
        mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, image.shape[1::-1])
        person_bboxes_top, person_bboxes_bottom = [], []

        if len(bboxes) > 0:
            person_points = [[int((bbox[2] + bbox[0]) / 2), int(bbox[3])] for bbox in bboxes]
            person_bboxes = list(zip(bboxes, person_points))

            person_bboxes_top = [pt for pt in person_bboxes if mask_top_court[pt[1][1]-1, pt[1][0]] == 1]
            person_bboxes_bottom = [pt for pt in person_bboxes if mask_bottom_court[pt[1][1] - 1, pt[1][0]] == 1]

            if filter_players:
                person_bboxes_top, person_bboxes_bottom = self.filter_players(person_bboxes_top, person_bboxes_bottom,
                                                                              matrix)
        return person_bboxes_top, person_bboxes_bottom

    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        # YOLO's confidence calibration runs much lower than the old Faster
        # R-CNN's for small/distant players (measured as low as ~0.44 for the
        # far player in a real broadcast frame, vs Faster R-CNN's usual >0.9)
        # - the top/bottom court-mask filter below does the real work of
        # rejecting non-players (ball kids, officials, crowd), so this can
        # stay low without letting false positives through.
        bboxes, probs = self.detect(image, person_min_score=config.PERSON_MIN_SCORE)
        return self.filter_top_bottom(image, inv_matrix, bboxes, probs, filter_players)

    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        """
        Leave one person at the top and bottom of the tennis court
        """
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        center_top_court = trans_kps[0][0]
        center_bottom_court = trans_kps[1][0]
        if len(person_bboxes_top) > 1:
            dists = [distance.euclidean(x[1], center_top_court) for x in person_bboxes_top]
            ind = dists.index(min(dists))
            person_bboxes_top = [person_bboxes_top[ind]]
        if len(person_bboxes_bottom) > 1:
            dists = [distance.euclidean(x[1], center_bottom_court) for x in person_bboxes_bottom]
            ind = dists.index(min(dists))
            person_bboxes_bottom = [person_bboxes_bottom[ind]]
        return person_bboxes_top, person_bboxes_bottom
    
    def track_players(self, frames, matrix_all, filter_players=False, batch_size=config.PERSON_INFER_BATCH_SIZE):
        min_len = min(len(frames), len(matrix_all))
        persons_top = [[] for _ in range(min_len)]
        persons_bottom = [[] for _ in range(min_len)]

        # frames without a homography have no meaningful court mask to filter
        # against, so they're skipped entirely (same as the original per-frame
        # loop, which never called detect() for them)
        valid_indices = [i for i in range(min_len) if matrix_all[i] is not None]

        for batch_start in tqdm(range(0, len(valid_indices), batch_size)):
            batch_indices = valid_indices[batch_start:batch_start + batch_size]
            batch_results = self.detect_batch([frames[i] for i in batch_indices])
            for idx, (bboxes, probs) in zip(batch_indices, batch_results):
                person_top, person_bottom = self.filter_top_bottom(frames[idx], matrix_all[idx], bboxes, probs,
                                                                      filter_players)
                persons_top[idx] = person_top
                persons_bottom[idx] = person_bottom
        return persons_top, persons_bottom


