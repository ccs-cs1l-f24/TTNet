import os
import sys
from collections import deque
import cv2
import numpy as np
import torch
import time
import json

sys.path.append('./')

from data_process.ttnet_video_loader_v2 import TTNet_Video_Loader_V2
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized
from pose.detect_pose import detect_pose

def process_video(configs):
    # Load the video and model
    video_loader = TTNet_Video_Loader_V2(configs.video_path, configs.input_size, configs.num_frames_sequence)
    frame_rate = video_loader.video_fps
    configs.device = torch.device("cuda:{}".format(configs.gpu_idx))
    model = create_model(configs)
    model.cuda()
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)
    model.eval()
    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1920, 1080
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize

    # For each frame
    with torch.no_grad():
        output_json_data = {
            "img_size": [w_original, h_original],
            "frames": []
        }

        for frame_cnt, frame_timestamp_ms, resized_imgs, orig_imgs in video_loader:
            img = orig_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0)
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)

            # Get ball position and whether it bounced
            t1 = time_synchronized()
            pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resized_imgs)
            t2 = time_synchronized()
            prediction_global, prediction_local, prediction_seg, prediction_events = post_processing(
                pred_ball_global, pred_ball_local, pred_events, pred_seg, configs.input_size[0],
                configs.thresh_ball_pos_mask, configs.seg_thresh, configs.event_thresh)
            prediction_ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2)
            ]

            # If it bounced, find position relative to table

            # Get the landmarks for left and right
            right_world_landmarks = detect_pose(img=img, blackout_left=True, frame_timestamp_ms=frame_timestamp_ms)
            left_world_landmarks = detect_pose(img=img, blackout_left=False, frame_timestamp_ms=frame_timestamp_ms)

            frame_data = {
                "frame": frame_cnt,
                "right_landmarks": right_world_landmarks,
                "left_landmarks": left_world_landmarks,
                "ball_pos": prediction_ball_final
            }
            output_json_data["frames"].append(frame_data)
    
        with open("../results/output_data.json", "w") as json_file:
            json.dump(output_json_data, json_file, indent=2)

    # Put in json array
    """
    {
        frames: 
        [
            {
                "left": 
                {
                    "landmark_0": (x, y),
                    "landmark_1": (x, y),
                    ...
                },
                "right": 
                {
                    "landmark_0": (x, y),
                    "landmark_1": (x, y),
                    ...
                },
                "ball_pos": (x, y) // Relative to table
            }
        ]
    }
    """

    # Download and send to Unity!


    pass

if __name__ == '__main__':
    configs = parse_configs()
    configs.video_path = "../dataset/test/videos/test_1_trimmed.mp4"
    configs.gpu_idx = 0
    configs.pretrained_path = "../checkpoints/ttnet.pth"
    configs.show_image = False
    configs.save_demo_output = True
    configs.no_seg = True

    process_video(configs=configs)