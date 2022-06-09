import os

from mmcv import FileClient
import mmcv
import pandas as pd
import numpy as np

data_root = '/home/myuser/resource/DAiSEE/DataSet/Train_112_scale1_frames'
data_root_val = '/home/myuser/resource/DAiSEE/DataSet/validation_112_scale1_frames'

if __name__ == '__main__':
    mmcv.use_backend('cv2')
    file_client = FileClient('disk')
    imgs = np.zeros([112, 112, 3, 1])
    means, stdevs = [], []
    for video_name in os.listdir(data_root):
        if video_name.endswith('aligned'):
            print("processing %s" % video_name)
            video_path = os.path.join(data_root, video_name)
            i = 0
            for frame_name in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame_name)
                img_bytes = file_client.get(frame_path)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                img = img[:, :, :, np.newaxis]
                imgs = np.concatenate((imgs, img), axis=3)
                i += 1
                if i >= 1:
                    break
    for video_name in os.listdir(data_root_val):
        if video_name.endswith('aligned'):
            print("processing %s" % video_name)
            video_path = os.path.join(data_root_val, video_name)
            i = 0
            for frame_name in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame_name)
                img_bytes = file_client.get(frame_path)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                img = img[:, :, :, np.newaxis]
                imgs = np.concatenate((imgs, img), axis=3)
                i += 1
                if i >= 1:
                    break

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    print(means, stdevs)
    '''
    openface2-scale1
    [97.7684514538713, 77.92614611871547, 73.06798767642202] [49.18597796201926, 41.639455424885575, 40.64126403977051]
    [102.21364698441803, 77.73569331768321, 72.60255504630771] [52.005905227796795, 40.494998096429484, 42.06624956519043]
    '''