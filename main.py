import os
import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from speed_estimator import get_ball_speed, get_shot_max_speed
from utils import scene_detect
import argparse
import torch

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
         ball_speed=None, shot_max_speed=None, draw_trace=False, trace=7):
    """
    :params
        frames: list of original images
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        ball_speed: list of instantaneous ball speed (km/h) per frame, or None to
            skip. Drawn next to the ball; changes every frame so it's hard to read
            while the ball is moving fast, but useful when paused/stepping frames.
        shot_max_speed: list of per-frame "current shot" peak speed (km/h, see
            speed_estimator.get_shot_max_speed), or None to skip. Shown as a fixed,
            stable HUD (readable while the ball is moving) alongside the ball_speed tag.
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images
    """
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices] 
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # draw ball trajectory
                if ball_track[i][0] is not None:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0] is not None:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball',
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                    if ball_speed is not None and ball_speed[i] is not None:
                        img_res = cv2.putText(img_res, '{:.0f} km/h'.format(ball_speed[i]),
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 30),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                          radius=0, color=(0, 0, 255), thickness=10)

                height, width, _ = img_res.shape

                # draw bounce in minimap
                if i in bounces and inv_mat is not None and ball_track[i][0] is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)

                minimap = court_img.copy()

                # draw persons
                persons = persons_top[i] + persons_bottom[i]
                for j, person in enumerate(persons):
                    person_bbox = list(person[0])
                    img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                            (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                    # transmit person point to minimap
                    person_point = list(person[1])
                    person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                    person_point = cv2.perspectiveTransform(person_point, inv_mat)
                    minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                                       radius=0, color=(255, 0, 0), thickness=80)

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap

                # fixed HUD (below the minimap): peak speed of the current bounce-to-bounce shot,
                # stable for the whole flight so it stays readable while the ball is moving
                if shot_max_speed is not None and shot_max_speed[i] is not None:
                    hud_x1 = width - 30 - width_minimap
                    hud_x2 = width - 30
                    hud_y1 = 30 + height_minimap + 10
                    hud_y2 = hud_y1 + 50
                    img_res = cv2.rectangle(img_res, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 0, 0), -1)
                    img_res = cv2.putText(img_res, '{:.0f} km/h'.format(shot_max_speed[i]),
                          org=(hud_x1 + 10, hud_y2 - 15),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.9,
                          thickness=2,
                          color=(0, 255, 255))

                imgs_res.append(img_res)

        else:    
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]] 
    return imgs_res        
 
def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()    


def _select_device(prefer_alt_gpu=True):
    """
    :params
        prefer_alt_gpu: whether a non-CUDA GPU backend (Apple MPS or Windows
            DirectML, via the optional torch_directml package) is a safe
            choice for this model. Both are newer, less mature backends
            where torchvision's detection ops (NMS/ROI-align) have been
            measured (MPS: ~67x slower than CPU on an Apple M5 Pro) or are
            suspected but UNVERIFIED (DirectML - no AMD/Windows hardware
            available to test) to run far slower than CPU - so callers must
            opt out for the person detector.
    """
    if torch.cuda.is_available():
        return 'cuda'
    if prefer_alt_gpu:
        if torch.backends.mps.is_available():
            return 'mps'
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except ImportError:
            pass
    return 'cpu'


def process_video(path_ball_track_model, path_court_model, path_bounce_model,
                   path_input_video, path_output_video, draw_trace=True, device=None,
                   detect_persons=True, progress_callback=None):
    """
    Run the full analysis pipeline on a video and write the annotated result.
    :params
        device: force a specific torch device for all models, or None to
            auto-select per model (see _select_device). If the resolved device
            is 'mps', person detection is forced to 'cpu' regardless (see
            _select_device's docstring for why).
        detect_persons: whether to run player detection at all. Faster R-CNN is
            the heaviest of the three models; set False to skip it entirely and
            speed up runs that only care about ball speed / court overlay.
        progress_callback: optional callable(str) invoked with a status message
            before each pipeline stage, so callers (e.g. a web UI) can show progress
    :return
        stats: dict summary (frame count, fps, bounce count, ball speed min/max/avg, devices used)
    """
    def report(message):
        print(message)
        if progress_callback:
            progress_callback(message)

    if device is None:
        ball_court_device = _select_device(prefer_alt_gpu=True)
        person_device = _select_device(prefer_alt_gpu=False)
    else:
        ball_court_device = device
        person_device = device
        if device == 'mps':
            person_device = 'cpu'
            report('not: oyuncu tespiti MPS\'te çok yavaş çalıştığı için CPU\'ya düşürüldü')

    report('video okunuyor')
    frames, fps = read_video(path_input_video)
    scenes = scene_detect(path_input_video)

    report('ball detection ({})'.format(ball_court_device))
    ball_detector = BallDetector(path_ball_track_model, ball_court_device)
    ball_track = ball_detector.infer_model(frames)

    report('court detection ({})'.format(ball_court_device))
    court_detector = CourtDetectorNet(path_court_model, ball_court_device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    if detect_persons:
        report('person detection ({})'.format(person_device))
        person_detector = PersonDetector(person_device)
        persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)
    else:
        persons_top = [[] for _ in frames]
        persons_bottom = [[] for _ in frames]

    report('bounce detection')
    bounce_detector = BounceDetector(path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    ball_speed = get_ball_speed(ball_track, homography_matrices, fps)
    shot_max_speed = get_shot_max_speed(ball_speed, bounces)

    report('video oluşturuluyor')
    imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
                    ball_speed=ball_speed, shot_max_speed=shot_max_speed, draw_trace=draw_trace)

    write(imgs_res, fps, path_output_video)

    valid_speeds = [s for s in ball_speed if s is not None]
    stats = {
        'num_frames': len(frames),
        'fps': fps,
        'num_bounces': len(bounces),
        'max_speed_kmh': max(valid_speeds) if valid_speeds else None,
        'avg_speed_kmh': (sum(valid_speeds) / len(valid_speeds)) if valid_speeds else None,
        'ball_court_device': ball_court_device,
        'person_device': person_device if detect_persons else None,
    }
    report('tamamlandı')
    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, required=True, help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', type=str, required=True, help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', type=str, required=True, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video', type=str, required=True, help='path to input video')
    parser.add_argument('--path_output_video', type=str, required=True, help='path to output video')
    args = parser.parse_args()

    for path_arg in ('path_ball_track_model', 'path_court_model', 'path_bounce_model', 'path_input_video'):
        path_value = getattr(args, path_arg)
        if not os.path.isfile(path_value):
            parser.error('{}: no such file: {}'.format(path_arg, path_value))

    stats = process_video(args.path_ball_track_model, args.path_court_model, args.path_bounce_model,
                           args.path_input_video, args.path_output_video, draw_trace=True)
    print(stats)





