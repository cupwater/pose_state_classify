'''
Author: Peng Bo
Date: 2022-08-11 09:50:26
LastEditTime: 2022-08-11 20:51:21
Description: 

'''
# coding: utf8
import os
import cv2
import numpy as np
import pdb

video_fps = 30
duration_window = 2*video_fps
smooth_window = 5
step_window = 15


def parse_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        # fps = cap.get(cv2.CAP_PROP_FPS)     # 返回视频的fps--帧率
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       # 返回视频的宽
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)     # 返回视频的高
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 返回视频的帧数
        print('width:', width, 'height:', height, 'num_frames:', num_frames)
        return width, height, num_frames
    return -1, -1, -1


def parse_labels(label_path):
    def parse_time(time_str):
        start_time, end_time = time_str.split('-')
        s_min, s_second = [int(a) for a in start_time.split(':')]
        e_min, e_second = [int(a) for a in end_time.split(':')]
        start_time = s_min*60 + s_second
        end_time = e_min*60 + e_second
        return start_time, end_time
    action_times = []
    with open(label_path) as fin:
        for line in fin.readlines()[1:]:
            # get the time, action and state_type
            t, a, s = line.strip().split(' ')
            action_times.append([a, parse_time(t), s])
    return action_times


def embedded_lms(raw_lms):
    reshaped_lms = raw_lms.reshape(raw_lms.shape[0], -1, 3)
    reshaped_lms = reshaped_lms[:, :7, :]
    reshaped_lms = reshaped_lms[:, :, :2].reshape(raw_lms.shape[0], -1)
    smooth_lms = reshaped_lms.reshape(-1, smooth_window, reshaped_lms.shape[1])
    smooth_lms = np.mean(smooth_lms, axis=1)
    return smooth_lms

# generate samples for training according the landmarks during particular during window


def generate_samples(lms, action_times):
    result_samples = []
    for a, [start_time, end_time], s in action_times:
        start_idx = start_time * video_fps - step_window if start_time > 0 else 0
        end_idx = end_time * video_fps - step_window
        for idx in range(start_idx, end_idx, step_window):
            select_lms = lms[idx:(idx+duration_window), :]
            if select_lms.shape[0] < duration_window:
                continue
            feature = embedded_lms(select_lms).reshape(-1)
            label = 1 if s == '状态' else 0
            result_samples.append([feature, label])

    return result_samples


if __name__ == "__main__":
    video_list = [
        "WFJ/WFJ01.mp4",
        "WFJ/WFJ02.mp4",
        "WFJ/WFJ03.mp4",
        "YW/YW01.mp4",
        "YW/YW02.mp4",
        "YW/YW03.mp4",
        "ZWR/ZWR01.mp4",
        "ZWR/ZWR02.mp4",
        "ZWR/ZWR03.mp4"
    ]

    lms_list = []
    action_times_lists = []
    for video_path in video_list:
        # width, height, num_frames = parse_video(video_path)
        # time_len = num_frames / video_fps

        lms_path = os.path.join(
            'pose_lms_mmpose', os.path.basename(video_path) + '_k2d.txt')
        lms_list.append(np.loadtxt(lms_path))

        labels_path = os.path.join('labels', os.path.basename(
            video_path).replace('mp4', 'txt'))
        action_times_lists.append(parse_labels(labels_path))

    all_samples = []
    for lms, action_times in zip(lms_list, action_times_lists):
        all_samples = all_samples + generate_samples(lms, action_times)
    features_list = [f for f, l in all_samples]
    labels_list = [l for f, l in all_samples]
    features = np.array(features_list).reshape(len(features_list), -1)
    metas = np.array(labels_list).reshape(-1, 1)

    v_max, v_min = np.max(features), np.min(features)
    normalize_features = ((features-v_min) / (v_max-v_min) - 0.5) * 4

    ratio = 0.7
    train_idxs = np.random.choice(
        range(metas.shape[0]), int(metas.shape[0]*0.7), replace=False)
    test_idxs = list(set(list(range(metas.shape[0]))) - set(train_idxs))

    train_features = normalize_features[train_idxs, :]
    train_metas = metas[train_idxs, :]

    test_features = normalize_features[test_idxs, :]
    test_metas = metas[test_idxs, :]

    result_folder = f'datasets/dur{duration_window}_step{step_window}_smo{smooth_window}_ratio{ratio}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    np.savetxt(os.path.join(result_folder, 'train_lms.txt'),
               train_features, fmt='%.3f')
    np.savetxt(os.path.join(result_folder, 'train_metas.txt'),
               train_metas,    fmt='%d')
    np.savetxt(os.path.join(result_folder, 'test_lms.txt'),
               test_features, fmt='%.3f')
    np.savetxt(os.path.join(result_folder, 'test_metas.txt'),
               test_metas,    fmt='%d')