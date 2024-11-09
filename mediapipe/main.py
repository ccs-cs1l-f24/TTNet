import mediapipe as mp
import cv2
import numpy as np
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.7
)

# Init video capture
cap = cv2.VideoCapture("../dataset/test/videos/test_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# JSON data collection
pose_data = {"left": [], "right": []}

def process_video(landmarker, blackout_left):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
    frame_cnt = 0
    width = size[0]
    data_key = "right" if blackout_left else "left"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Apply left or right blackout
        if blackout_left:
            frame[:, :width // 2] = 0  # Blackout left half
        else:
            frame[:, width // 2:] = 0  # Blackout right half

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Check if landmarks are detected
        if pose_result.pose_world_landmarks:
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_result)
            to_window = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image with pose", to_window)

            # Collect landmarks for JSON export
            frame_landmarks = {
                "frame": frame_cnt,
                "landmarks": []
            }

            for idx, pose_world_landmarks in enumerate(pose_result.pose_world_landmarks):
                landmarks = [{"name": f"landmark_{i}", "x": landmark.x, "y": landmark.y, "z": landmark.z}
                             for i, landmark in enumerate(pose_world_landmarks)]
                frame_landmarks["landmarks"] = landmarks

            pose_data[data_key].append(frame_landmarks)

        frame_cnt += 1

        if cv2.waitKey(1) == ord('q'):
            break

# Run with left half blacked out
with PoseLandmarker.create_from_options(options) as landmarker:
    print("Processing with left half blacked out")
    process_video(landmarker, blackout_left=True)

# Run with right half blacked out
with PoseLandmarker.create_from_options(options) as landmarker:
    print("Processing with right half blacked out")
    process_video(landmarker, blackout_left=False)

cap.release()
cv2.destroyAllWindows()

# Export collected pose data to JSON with the "left" and "right" structure
with open("pose_landmarks.json", "w") as json_file:
    json.dump(pose_data, json_file, indent=2)
