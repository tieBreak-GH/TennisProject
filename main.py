import os
import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from speed_estimator import get_ball_speed, get_shot_max_speed
from rally_analyzer import analyze_rallies, select_highlights
from utils import SceneCutDetector, ThreadedFrameReader, ThreadedFrameWriter
import argparse
import time
import torch

import config

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

_MINIMAP_WIDTH = 166
_MINIMAP_HEIGHT = 350

_SERVE_LINE_CALL_LABELS = {
    'in': 'SERVİS - İÇERİ',
    'out': 'SERVİS - DIŞARI',
    'belirsiz': 'SERVİS - BELİRSİZ',
}


def render_frame(img_res, i, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
                  ball_speed, shot_max_speed, serve_labels, court_img, draw_trace, trace):
    """
    Render one frame's overlays (ball trace/speed/serve label, court
    keypoints, minimap with bounce marks + player dots, HUD). Extracted from
    the old render_output's per-frame loop body so render_streaming can call
    it once per decoded frame instead of building a second full-length list
    of rendered frames.
    :params
        img_res: a mutable working copy of the raw frame (caller's responsibility
            to .copy() the decoded frame before calling)
        court_img: minimap render state (accumulates bounce marks) - the caller
            must carry this across frames within the same drawable scene and
            reset it (via get_court_img()) whenever a new drawable scene begins
    :return
        (img_res, court_img): the rendered frame and the (possibly mutated) minimap state
    """
    inv_mat = homography_matrices[i]

    # draw ball trajectory
    if ball_track[i][0] is not None:
        if draw_trace:
            for j in range(0, trace):
                if i-j >= 0:
                    if ball_track[i-j][0] is not None:
                        draw_x = int(ball_track[i-j][0])
                        draw_y = int(ball_track[i-j][1])
                        img_res = cv2.circle(img_res, (draw_x, draw_y),
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

        serve_label = serve_labels.get(i) if serve_labels is not None else None
        if serve_label:
            img_res = cv2.putText(img_res, serve_label,
                  org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 52),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=0.8,
                  thickness=2,
                  color=(0, 255, 255))

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

    minimap = cv2.resize(minimap, (_MINIMAP_WIDTH, _MINIMAP_HEIGHT))
    img_res[30:(30 + _MINIMAP_HEIGHT), (width - 30 - _MINIMAP_WIDTH):(width - 30), :] = minimap

    # fixed HUD (below the minimap): peak speed of the current bounce-to-bounce shot,
    # stable for the whole flight so it stays readable while the ball is moving
    if shot_max_speed is not None and shot_max_speed[i] is not None:
        hud_x1 = width - 30 - _MINIMAP_WIDTH
        hud_x2 = width - 30
        hud_y1 = 30 + _MINIMAP_HEIGHT + 10
        hud_y2 = hud_y1 + 50
        img_res = cv2.rectangle(img_res, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 0, 0), -1)
        img_res = cv2.putText(img_res, '{:.0f} km/h'.format(shot_max_speed[i]),
              org=(hud_x1 + 10, hud_y2 - 15),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=0.9,
              thickness=2,
              color=(0, 255, 255))

    return img_res, court_img


def _open_writer(path, fps, width, height):
    # avc1 (H.264) instead of DIVX (MPEG-4 Part 2): browsers, including
    # Chrome desktop and mobile, cannot play DIVX/FMP4 in an HTML5 <video>
    # element, which broke the web UI's video preview.
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    if not writer.isOpened():
        # cv2.VideoWriter doesn't raise on a missing/broken encoder - it
        # silently produces an empty or corrupt file instead, which then
        # surfaces much later as a confusing "video won't play" report.
        raise RuntimeError(
            "Video yazici acilamadi (avc1/H.264 codec). OpenCV/FFmpeg "
            "kurulumunuzda H.264 encoder eksik olabilir. Hedef dosya: {}".format(path))
    return writer


def analyze_streaming(path_input_video, ball_detector, court_detector, person_detector,
                       detect_persons, ball_batch_size, person_batch_size, report):
    """
    Pass 1 of the streaming pipeline: decode the video once, frame by frame,
    driving ball/court/person/scene-cut detection inline, and keep only
    their lightweight per-frame metadata (never the whole video's pixels at
    once). Peak memory is bounded by a small batch window instead of video
    length.

    Scene cuts (SceneCutDetector) are detected on the fly from this same
    decode instead of PySceneDetect's own separate full-video pass, so the
    video is decoded twice overall (this pass + render_streaming's) instead
    of three times.

    Court homography reproduces court_detector.infer_model's exact
    broadcast-per-scene semantics: it probes up to config.COURT_MAX_PROBE_FRAMES
    frames at the start of each scene, then assigns whichever result it found
    (even to the probe frames themselves) to the entire scene - so this
    buffers just that small probe window (default 5 frames) per scene until
    resolved, then backfills. Person detection on the buffered probe-window
    frames is deferred until the scene's homography resolves, then run
    batched on the backlog; frames afterward reuse the resolved homography
    immediately.

    Frame decode happens on a background thread (ThreadedFrameReader) so
    decoding frame i+1 overlaps with running the ball/court/person models on
    frame i, instead of the two taking turns.
    :return
        (ball_track, homography_matrices, kps_court, persons_top, persons_bottom, fps, num_frames, scenes)
    """
    reader = ThreadedFrameReader(path_input_video)
    fps = reader.fps
    frame_count_hint = reader.frame_count_hint

    ball_track = [(None, None), (None, None)]
    ball_window = []
    ball_prev_pred = [None, None]
    ball_scale_x = ball_scale_y = None

    court_width, court_height = 640, 360
    court_scale_x = court_scale_y = None

    homography_matrices = []
    kps_court = []
    persons_top = []
    persons_bottom = []

    scene_cut_detector = SceneCutDetector()
    scenes = []
    scene_start = 0
    scene_matrix, scene_points, scene_probe_count = None, None, 0
    scene_pending = []
    person_pending = []

    def flush_person(pending, matrix):
        if not pending or matrix is None or not detect_persons:
            return
        results = person_detector.detect_batch([f for _, f in pending])
        for (idx, frame), (bboxes, probs) in zip(pending, results):
            top, bottom = person_detector.filter_top_bottom(frame, matrix, bboxes, probs)
            persons_top[idx] = top
            persons_bottom[idx] = bottom

    def resolve_scene():
        for idx, _ in scene_pending:
            homography_matrices[idx] = scene_matrix
            kps_court[idx] = scene_points
        flush_person(scene_pending, scene_matrix)
        scene_pending.clear()

    def flush_ball(final=False):
        nonlocal ball_window, ball_prev_pred
        if len(ball_window) < 3:
            return
        predictions, ball_prev_pred = ball_detector.infer_batch(ball_window, ball_prev_pred, ball_scale_x, ball_scale_y)
        ball_track.extend(predictions)
        ball_window = [] if final else ball_window[-2:]

    i = 0
    try:
        for frame in reader:
            if ball_scale_x is None:
                orig_height, orig_width = frame.shape[:2]
                ball_scale_x = orig_width / ball_detector.width
                ball_scale_y = orig_height / ball_detector.height
                court_scale_x = orig_width / court_width
                court_scale_y = orig_height / court_height

            homography_matrices.append(None)
            kps_court.append(None)
            persons_top.append([])
            persons_bottom.append([])

            # ball: continuous sliding window, independent of scene boundaries
            ball_window.append(cv2.resize(frame, (ball_detector.width, ball_detector.height)))
            if len(ball_window) >= ball_batch_size + 2:
                flush_ball()

            # scene boundary: a detected cut closes the current scene here and
            # opens a new one, instead of walking a precomputed scene list
            if scene_cut_detector.is_cut(i, frame):
                if scene_pending:
                    resolve_scene()
                scenes.append((scene_start, i))
                scene_start = i
                scene_matrix, scene_points, scene_probe_count = None, None, 0

            # court + person
            if scene_matrix is None and scene_probe_count < config.COURT_MAX_PROBE_FRAMES:
                m, p = court_detector.infer_frame(frame, court_scale_x, court_scale_y)
                scene_probe_count += 1
                scene_pending.append((i, frame))
                if m is not None:
                    scene_matrix, scene_points = m, p
                if scene_matrix is not None or scene_probe_count >= config.COURT_MAX_PROBE_FRAMES:
                    resolve_scene()
            else:
                homography_matrices[i] = scene_matrix
                kps_court[i] = scene_points
                if detect_persons and scene_matrix is not None:
                    person_pending.append((i, frame))
                    if len(person_pending) >= person_batch_size:
                        flush_person(person_pending, scene_matrix)
                        person_pending = []

            if i % 20 == 0:
                report('video analiz ediliyor', min(0.02 + 0.68 * (i / frame_count_hint), 0.70))

            i += 1

        flush_ball(final=True)
        if person_pending:
            flush_person(person_pending, scene_matrix)
        if scene_pending:
            resolve_scene()
        scenes.append((scene_start, i))
    finally:
        reader.close()

    return ball_track, homography_matrices, kps_court, persons_top, persons_bottom, fps, i, scenes


def render_streaming(path_input_video, path_output_video, scenes, bounces, ball_track, homography_matrices,
                      kps_court, persons_top, persons_bottom, ball_speed, shot_max_speed, serve_labels,
                      rallies, num_frames, fps, draw_trace, trace, generate_highlights, highlights_top_n,
                      highlights_dir, trim_dead_time, rallies_only_path, report):
    """
    Pass 2 of the streaming pipeline: decode the video a second time, frame
    by frame, render each with Pass 1's metadata and write it straight to
    the output VideoWriter (and any rallies-only/highlight writer whose
    window it falls in), discarding it immediately - no second full-length
    frame list, no third decode for highlight/rallies-only clips.

    Decode (ThreadedFrameReader) and the main output write (ThreadedFrameWriter)
    each run on their own background thread, so decode/render/encode overlap
    instead of taking turns on one thread. The rallies-only/highlight writers
    stay synchronous - they're opened/closed dynamically as rally windows are
    entered and left, and write far less often than the main output.
    :return
        (rallies_only_video, highlight_clips) - same shape as the old
        write_rallies_only/write_highlights return values
    """
    eps = 1e-15
    is_track = [x is not None for x in homography_matrices]
    scene_drawable = []
    for s, e in scenes:
        sum_track = sum(is_track[s:e])
        len_track = e - s
        scene_drawable.append(sum_track/(len_track+eps) > 0.5)

    highlights = select_highlights(rallies, top_n=highlights_top_n) if generate_highlights else []

    out_dir_highlights = highlights_dir or os.path.join(
        os.path.dirname(os.path.abspath(path_output_video)), 'highlights')
    out_path_rallies_only = rallies_only_path or os.path.join(
        os.path.dirname(os.path.abspath(path_output_video)), 'rallies_only.mp4')

    reader = ThreadedFrameReader(path_input_video)
    main_writer = None
    rallies_writer = None
    highlight_writer = None
    highlight_writer_info = None
    highlight_clips = []
    rally_ptr = 0
    highlight_ptr = 0

    scene_ptr = 0
    scene_start, scene_end = scenes[0]
    court_img = get_court_img() if scene_drawable[0] else None

    i = 0
    try:
        for frame in reader:
            if main_writer is None:
                height, width = frame.shape[:2]
                main_writer = ThreadedFrameWriter(_open_writer(path_output_video, fps, width, height))

            if i == scene_end and scene_ptr + 1 < len(scenes):
                scene_ptr += 1
                scene_start, scene_end = scenes[scene_ptr]
                court_img = get_court_img() if scene_drawable[scene_ptr] else None

            if scene_drawable[scene_ptr]:
                img_res = frame.copy()
                img_res, court_img = render_frame(img_res, i, bounces, ball_track, homography_matrices, kps_court,
                                                   persons_top, persons_bottom, ball_speed, shot_max_speed,
                                                   serve_labels, court_img, draw_trace, trace)
            else:
                img_res = frame

            main_writer.write(img_res)

            if trim_dead_time:
                while rally_ptr < len(rallies) and i >= rallies[rally_ptr]['end_frame']:
                    rally_ptr += 1
                if rally_ptr < len(rallies) and rallies[rally_ptr]['start_frame'] <= i < rallies[rally_ptr]['end_frame']:
                    if rallies_writer is None:
                        os.makedirs(os.path.dirname(os.path.abspath(out_path_rallies_only)), exist_ok=True)
                        rallies_writer = _open_writer(out_path_rallies_only, fps, width, height)
                    rallies_writer.write(img_res)

            if generate_highlights:
                if highlight_ptr < len(highlights) and i >= highlights[highlight_ptr]['rally']['end_frame']:
                    if highlight_writer is not None:
                        highlight_writer.release()
                        highlight_clips.append(highlight_writer_info)
                        highlight_writer = None
                    highlight_ptr += 1
                if highlight_ptr < len(highlights):
                    h = highlights[highlight_ptr]
                    r = h['rally']
                    if r['start_frame'] <= i < r['end_frame']:
                        if highlight_writer is None:
                            os.makedirs(out_dir_highlights, exist_ok=True)
                            reason_tag = '_'.join(h['reasons'])
                            path = os.path.join(out_dir_highlights,
                                                 'highlight_rally{:02d}_{}.mp4'.format(r['rally_no'], reason_tag))
                            highlight_writer = _open_writer(path, fps, width, height)
                            highlight_writer_info = {'rally_no': r['rally_no'], 'reasons': h['reasons'], 'path': path}
                        highlight_writer.write(img_res)

            if i % 20 == 0:
                report('video oluşturuluyor ve yazılıyor', 0.75 + 0.23 * (i / max(num_frames, 1)))

            i += 1

        if highlight_writer is not None:
            highlight_writer.release()
            highlight_clips.append(highlight_writer_info)
            highlight_writer = None
    finally:
        reader.close()
        if main_writer is not None:
            main_writer.release()
        if rallies_writer is not None:
            rallies_writer.release()
        if highlight_writer is not None:
            highlight_writer.release()

    rallies_only_video = out_path_rallies_only if rallies_writer is not None else None
    return rallies_only_video, highlight_clips


def _select_device(prefer_alt_gpu=True):
    """
    :params
        prefer_alt_gpu: whether to try a non-CUDA GPU backend (Apple MPS, or
            Windows DirectML via the optional torch_directml package) before
            falling back to CPU. DirectML remains UNVERIFIED (no AMD/Windows
            hardware available to test); callers that need a guaranteed-safe
            fallback can pass False.
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


def _infer_batch_size(device, gpu_batch_size):
    """
    Batching frames into one model forward pass only pays off when per-call
    kernel-launch overhead dominates, which is the well-established case on
    CUDA. Measured on this dev machine (Apple Silicon, no CUDA): batch_size=16
    was ~4x SLOWER than sequential on CPU, and slower than plain CPU
    sequential on MPS too (both backends apparently lack the kernel-launch
    overhead that batching is meant to amortize, or don't parallelize this
    model's ops well across a batch dim). So default to sequential
    (batch_size=1, the original per-frame behavior) everywhere except CUDA.
    """
    return gpu_batch_size if str(device) == 'cuda' else 1


def process_video(path_ball_track_model, path_court_model, path_bounce_model,
                   path_input_video, path_output_video, draw_trace=True, device=None,
                   detect_persons=True, progress_callback=None,
                   generate_highlights=False, highlights_dir=None, highlights_top_n=3,
                   trim_dead_time=False, rallies_only_path=None):
    """
    Run the full analysis pipeline on a video and write the annotated result.
    :params
        device: force a specific torch device for all models, or None to
            auto-select per model (see _select_device).
        detect_persons: whether to run player detection at all. Set False to
            skip it entirely and speed up runs that only care about ball
            speed / court overlay.
        progress_callback: optional callable(message, fraction, eta_seconds)
            invoked periodically (every ~20 frames within each pass, plus at
            each pipeline phase boundary), so callers (e.g. a web UI) can
            show a real, frame-driven progress bar. fraction is 0..1.
            eta_seconds is the estimated time left based on elapsed time /
            fraction so far, or None on the first call (no elapsed time to
            extrapolate from yet).
        generate_highlights: whether to also cut short clips for the
            fastest-shot / longest rallies (see rally_analyzer.select_highlights).
        highlights_dir: directory to write highlight clips into, or None to
            use a 'highlights' subfolder next to path_output_video.
        highlights_top_n: how many rallies to pick per highlight criterion.
        trim_dead_time: whether to also write a single video that
            concatenates only the rally windows (see render_streaming),
            skipping the non-rally "dead time" between points.
        rallies_only_path: output path for the dead-time-trimmed video, or
            None to use 'rallies_only.mp4' next to path_output_video.
    :return
        stats: dict summary (frame count, fps, bounce count, ball speed min/max/avg, devices used)
    """
    start_time = time.time()

    def report(message, fraction):
        elapsed = time.time() - start_time
        eta = elapsed * (1 - fraction) / fraction if fraction > 0 else None
        print('[{:.0%}] {}'.format(fraction, message))
        if progress_callback:
            progress_callback(message, fraction, eta)

    if device is None:
        ball_court_device = _select_device(prefer_alt_gpu=True)
        # YOLO (unlike the old Faster R-CNN person detector) was benchmarked
        # at 11ms/frame on MPS vs 16ms/frame on CPU - no MPS performance pit,
        # so it can share the same device-selection policy as ball/court.
        person_device = _select_device(prefer_alt_gpu=True)
    else:
        ball_court_device = device
        person_device = device

    ball_detector = BallDetector(path_ball_track_model, ball_court_device)
    court_detector = CourtDetectorNet(path_court_model, ball_court_device)
    person_detector = PersonDetector(person_device) if detect_persons else None

    report('video analiz ediliyor ({})'.format(ball_court_device), 0.02)
    ball_track, homography_matrices, kps_court, persons_top, persons_bottom, fps, num_frames, scenes = analyze_streaming(
        path_input_video, ball_detector, court_detector, person_detector, detect_persons,
        _infer_batch_size(ball_court_device, config.BALL_INFER_BATCH_SIZE),
        _infer_batch_size(person_device, config.PERSON_INFER_BATCH_SIZE),
        report)

    report('sekme tespiti', 0.72)
    bounce_detector = BounceDetector(path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    ball_speed = get_ball_speed(ball_track, homography_matrices, fps, bounces)
    shot_max_speed = get_shot_max_speed(ball_speed, bounces)

    rallies = analyze_rallies(ball_track, bounces, homography_matrices, ball_speed, fps)
    serve_labels = {}
    for r in rallies:
        for s in r['shots']:
            if s['is_serve']:
                label = _SERVE_LINE_CALL_LABELS.get(s['line_call'], 'SERVİS')
                for f in range(s['start_frame'], s['end_frame']):
                    serve_labels[f] = label

    report('video oluşturuluyor ve yazılıyor', 0.75)
    rallies_only_video, highlight_clips = render_streaming(
        path_input_video, path_output_video, scenes, bounces, ball_track, homography_matrices, kps_court,
        persons_top, persons_bottom, ball_speed, shot_max_speed, serve_labels, rallies, num_frames, fps,
        draw_trace, 7, generate_highlights, highlights_top_n, highlights_dir, trim_dead_time,
        rallies_only_path, report)

    valid_speeds = [s for s in ball_speed if s is not None]
    stats = {
        'num_frames': num_frames,
        'fps': fps,
        'num_bounces': len(bounces),
        'num_rallies': len(rallies),
        'rallies': rallies,
        'ball_speed': ball_speed,
        'highlight_clips': highlight_clips,
        'rallies_only_video': rallies_only_video,
        'max_speed_kmh': max(valid_speeds) if valid_speeds else None,
        'avg_speed_kmh': (sum(valid_speeds) / len(valid_speeds)) if valid_speeds else None,
        'ball_court_device': ball_court_device,
        'person_device': person_device if detect_persons else None,
    }
    report('tamamlandı', 1.0)
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





