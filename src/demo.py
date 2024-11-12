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


def process_video(configs):
    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)

    # Set up video writer
    frame_rate = video_loader.video_fps

    if configs.save_demo_output:
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for the video format
        video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (1920, 1080))

    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    # Load model
    model = create_model(configs)
    model.cuda()
    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)
    model.eval()

    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1920, 1080
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize

    with torch.no_grad():
        for count, resized_imgs in video_loader:
            # Process the current frame
            img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0), (w_original, h_original))
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)

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

            # Display middle frame's prediction if queue is filled
            if len(queue_frames) == middle_idx + 1:
                frame_pred_infor = queue_frames.popleft()
                ball_pos = frame_pred_infor['ball']
                seg_img = frame_pred_infor['seg'].astype(np.uint8)
                if configs.no_seg:
                    seg_img[:] = 0
                seg_img = cv2.resize(seg_img, (w_original, h_original))
                ploted_img = plot_detection(img, ball_pos, seg_img, prediction_events)
                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)

                # Write frame to video
                if configs.save_demo_output:
                    video_writer.write(ploted_img)

                # Show image if desired
                if configs.show_image:
                    cv2.imshow('ploted_img', ploted_img)
                    cv2.waitKey(10)

            frame_pred_infor = {
                'seg': prediction_seg,
                'ball': prediction_ball_final
            }
            queue_frames.append(frame_pred_infor)

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s. Ball Pos: {}'.format(frame_idx, t2 - t1, prediction_ball_final))

    # Release the video writer
    if configs.save_demo_output:
        video_writer.release()
        print("Video has been written to:", output_path)


import cv2

def plot_detection(img, ball_pos, seg_img, events):
    """Show the predicted information in the image."""

    img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    
    event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add a green border if bounce is above 0.7
    if events[0] > 0.2:
        border_thickness = 10
        img = cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    
    return img


if __name__ == '__main__':
    configs = parse_configs()
    configs.video_path = "../dataset/test/videos/test_1_trimmed.mp4"
    configs.gpu_idx = 0
    configs.pretrained_path = "../checkpoints/ttnet.pth"
    configs.show_image = False
    configs.save_demo_output = True
    configs.no_seg = True
    process_video(configs=configs)
