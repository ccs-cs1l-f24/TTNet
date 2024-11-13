"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.10
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script creates the video loader for testing with an input video
"""

import os
from collections import deque

import cv2
import numpy as np


class TTNet_Video_Loader_V2:
    """The loader for demo with a video input"""

    def __init__(self, video_path, input_size=(320, 128), num_frames_sequence=9):
        assert os.path.isfile(video_path), "No video at {}".format(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = input_size[0]
        self.height = input_size[1]
        self.count = 0
        self.frame_timestamp_ms = 0
        self.num_frames_sequence = num_frames_sequence
        print('Length of the video: {:d} frames'.format(self.video_num_frames))

        self.images_sequence = deque(maxlen=num_frames_sequence)
        self.orig_images_sequence = deque(maxlen=num_frames_sequence)
        self.get_first_images_sequence()

    def get_first_images_sequence(self):
        # Load (self.num_frames_sequence - 1) images
        while (self.count < self.num_frames_sequence):
            self.count += 1
            ret, frame = self.cap.read()  # BGR
            assert ret, 'Failed to load frame {:d}'.format(self.count)
            self.images_sequence.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height)))
            self.orig_images_sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self) - 1:
            raise StopIteration

        # Read image
        ret, frame = self.cap.read()  # BGR
        assert ret, 'Failed to load frame {:d}'.format(self.count)
        self.frame_timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        self.images_sequence.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height)))
        self.orig_images_sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        resized_imgs = np.dstack(self.images_sequence)  # (128, 320, 27)
        # Transpose (H, W, C) to (C, H, W) --> fit input of TTNet model
        resized_imgs = resized_imgs.transpose(2, 0, 1)  # (27, 128, 320)

        orig_imgs = np.dstack(self.orig_images_sequence) # (1080, 1920, 27)
        orig_imgs = orig_imgs.transpose(2, 0, 1) # (27, 1080, 1920)

        return self.count, self.frame_timestamp_ms, resized_imgs, orig_imgs

    def __len__(self):
        return self.video_num_frames - self.num_frames_sequence + 1  # number of sequences
