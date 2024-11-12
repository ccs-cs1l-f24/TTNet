import os
import sys
from collections import deque
import cv2
import numpy as np
import torch
import time

sys.path.append('./')

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized

def process_video():

    # Load the video

    # For each frame

    # Get ball position
    # Detect whether it bounced => Find position relative to table

    # Get the landmarks for left and right

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