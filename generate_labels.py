'''
Author: Peng Bo
Date: 2022-08-11 09:50:26
LastEditTime: 2022-08-12 22:13:01
Description: 

'''
# coding: utf8
import os
import cv2
import numpy as np
import pdb

video_fps = 30
duration_window = int(1*video_fps)
pool_window = 5
step_window = 5


def parse_video(video_path, lms):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)     # 返回视频的fps--帧率
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       # 返回视频的宽
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)     # 返回视频的高
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 返回视频的帧数
        print('width:', width, 'height:', height, 'num_frames:', num_frames)
        idx = 0
        while(True):
            #capture frame-by-frame
            _, frame = cap.read()
            seconds = cap.get(cv2.CAP_PROP_POS_MSEC) // 1000
            print(f'second: {seconds}, parse: {idx//30}')
            lm = lms[idx].reshape(17, 3)
            idx += 1
            for i in range(7):
                cv2.circle(frame, (int(lm[i,0]), int(lm[i,1])), 2, (255,0,0), 2)
            #display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) &0xFF ==ord('q'):  #按q键退出
                break
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
    reshaped_lms = reshaped_lms[:, :7, :2].reshape(raw_lms.shape[0], -1)
    return reshaped_lms.reshape(-1)

    # smooth_lms = reshaped_lms.reshape(-1, pool_window, reshaped_lms.shape[1])
    # smooth_lms = np.mean(smooth_lms, axis=1)
    # diff_lms = smooth_lms[1:,:] -  smooth_lms[:-1,:]

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
            feature = embedded_lms(select_lms)
            label = 1 if s == '状态' else 0
            result_samples.append([feature, label])

    return result_samples


if __name__ == "__main__":
    video_list = [
        "../升降显示器视频数据集/WFJ/WFJ01.mp4",
        "../升降显示器视频数据集/WFJ/WFJ02.mp4",
        "../升降显示器视频数据集/WFJ/WFJ03.mp4",
        "../升降显示器视频数据集/YW/YW01.mp4",
        "../升降显示器视频数据集/YW/YW02.mp4",
        "../升降显示器视频数据集/YW/YW03.mp4",
        "../升降显示器视频数据集/ZWR/ZWR01.mp4",
        "../升降显示器视频数据集/ZWR/ZWR02.mp4",
        "../升降显示器视频数据集/ZWR/ZWR03.mp4"
    ]

    lms_list = []
    action_times_lists = []
    for video_path in video_list:
        lms_path = os.path.join(
            'mmpose_lms', os.path.basename(video_path) + '_k2d.txt')
        lms = np.loadtxt(lms_path)
        lms_list.append(lms)
        labels_path = os.path.join('labels', os.path.basename(
            video_path).replace('mp4', 'txt'))
        action_times_lists.append(parse_labels(labels_path))
        # width, height, num_frames = parse_video(video_path, lms)
        # time_len = num_frames / video_fps

    all_samples = []
    for lms, action_times in zip(lms_list, action_times_lists):
        all_samples = all_samples + generate_samples(lms, action_times)
    features_list = [f for f, l in all_samples]
    labels_list = [l for f, l in all_samples]
    features = np.array(features_list).reshape(len(features_list), -1)
    metas = np.array(labels_list).reshape(-1, 1)

    def normalize_matrix(features):
        normalize_features = features.copy()
        for col in range(features.shape[1]):
            v_max, v_min = np.max(features[:,col]), np.min(features[:,col])
            normalize_features[:,col] = ((features[:,col]-v_min) / (v_max-v_min) - 0.5) * 4
        return normalize_features
    normalize_features = normalize_matrix(features)

    ratio = 0.7
    train_idxs = np.random.choice(
        range(metas.shape[0]), int(metas.shape[0]*0.7), replace=False)
    test_idxs = list(set(list(range(metas.shape[0]))) - set(train_idxs))
    train_features = normalize_features[train_idxs, :]
    train_metas = metas[train_idxs, :]
    test_features = normalize_features[test_idxs, :]
    test_metas = metas[test_idxs, :]



    result_folder = f'datasets/dur{duration_window}_step{step_window}_smo{pool_window}_ratio{ratio}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    np.savetxt(os.path.join(result_folder, 'lms.txt'),
               normalize_features, fmt='%.3f')
    np.savetxt(os.path.join(result_folder, 'metas.txt'),
               metas,    fmt='%d')
    np.savetxt(os.path.join(result_folder, 'train_lms.txt'),
               train_features, fmt='%.3f')
    np.savetxt(os.path.join(result_folder, 'train_metas.txt'),
               train_metas,    fmt='%d')
    np.savetxt(os.path.join(result_folder, 'test_lms.txt'),
               test_features, fmt='%.3f')
    np.savetxt(os.path.join(result_folder, 'test_metas.txt'),
               test_metas,    fmt='%d')
