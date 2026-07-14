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


