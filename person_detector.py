import cv2
import numpy as np
from ultralytics import YOLO
from court_reference import CourtReference

import config


class PersonDetector():
    def __init__(self, device='cpu'):
        self.detection_model = YOLO('yolo11n.pt')
        self.device = device
        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)


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

    def filter_top_bottom(self, image, inv_matrix, bboxes, probs):
        matrix = cv2.invert(inv_matrix)[1]
        mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, image.shape[1::-1])
        mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, image.shape[1::-1])
        person_bboxes_top, person_bboxes_bottom = [], []

        if len(bboxes) > 0:
            h, w = mask_top_court.shape[:2]
            person_points = [[int((bbox[2] + bbox[0]) / 2), int(bbox[3])] for bbox in bboxes]
            person_bboxes = list(zip(bboxes, person_points))

            for pt in person_bboxes:
                x = np.clip(pt[1][0], 0, w - 1)
                y = np.clip(pt[1][1] - 1, 0, h - 1)
                if mask_top_court[y, x] == 1:
                    person_bboxes_top.append(pt)
                if mask_bottom_court[y, x] == 1:
                    person_bboxes_bottom.append(pt)

        return person_bboxes_top, person_bboxes_bottom

