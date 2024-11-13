import mediapipe as mp
import cv2
import numpy as np
import json
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MP_MODEL_PATH = os.path.join(CURRENT_DIR, "pose_landmarker_heavy.task")

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.8,
    min_pose_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

def detect_pose(img: np.ndarray, blackout_left: bool, frame_timestamp_ms: int):
    """ Detect human pose on an image
    :param img: (1080, 1920, 3) in RGB
    :param blackout_left: Should the left side be blacked out (otherwise right is blacked out)
    :param frame_timestamp_ms: The timestamp to the nearest millisecond
    :return: array of landmarks (which are relative to the hips)
    """

    height, width, channels = img.shape

    with PoseLandmarker.create_from_options(options) as landmarker:
        if blackout_left:
            img[:, :width//2] = 0
        else:
            img[:, width//2:] = 0

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if pose_result.pose_world_landmarks:
            world_landmarks = [{"name": f"landmark_{i}", "x": landmark.x, "y": landmark.y, "z": landmark.z, "visibility": landmark.visibility}
                                for i, landmark in enumerate(pose_result.pose_world_landmarks[0])]

            return world_landmarks

    print("ERROR: No landmarks detected")
    return []
        