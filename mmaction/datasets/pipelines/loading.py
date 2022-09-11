import os.path as osp

import cv2
import mmcv
import numpy as np
from mmcv.fileio import FileClient

from ..builder import PIPELINES


@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index".

    Args:
        num_wins (int): Frames of each sampled output clip.
        alpha (int): Ratio of the window size and the stride size.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        gamma (int): Sample rate to select the frames of slide windows.
    """

    def __init__(self,
                 num_wins,
                 alpha=3,
                 test_mode=False,
                 gamma=1):
        self.num_wins = num_wins
        self.alpha = alpha
        self.test_mode = test_mode
        self.gamma = gamma

    def _sort_by_binarytree(self, w_start, w_end):
        result = []
        k = (int)(np.log2(w_end - w_start + 2))
        for i in range(k):
            node_num = 2 ** i
            stride = (w_end - w_start) / (2 * node_num)
            for j in range(1, node_num + 1):
                node_pos = (int)(w_start + (2 * j - 1) * stride)
                result.append(node_pos)

        return result

    def _sample_frames(self, num_frames, clip_start_idx):
        """Choose frames for the video sampled in the slide windows.

        Args:
            num_frames (int): Total number of frame in the video.
            clip_start_idx (int): Index of the frame in the first window.

        Returns:
            seq(list): an instance sampled from the video, [[f1, f2, ..., fnum_wins]].
        """

        ds_frame_num = num_frames // self.gamma
        w_stride = (int)(ds_frame_num / (self.alpha + self.num_wins - 1))
        w_size = self.alpha * w_stride

        # build windows with frame positions in 2-path tree
        first_window = self._sort_by_binarytree(0, w_size - 1)
        windows = []
        for k in range(self.num_wins):
            win_inds = np.array(first_window) + k * w_stride
            windows.append(win_inds.tolist())

        # Sample on the video to generate an instance
        instances = np.array(windows).T
        ds_video = np.array([i * self.gamma for i in range(ds_frame_num)])
        sampled_instance = ds_video[instances[clip_start_idx]][0:self.num_wins]

        return np.array([sampled_instance])

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        clip_start_idx = results['clip_start_idx']
        frame_inds = self._sample_frames(total_frames, clip_start_idx)
        frame_inds = np.concatenate(frame_inds)
        frame_inds = frame_inds.reshape((-1, self.num_wins))
        frame_inds = np.mod(frame_inds, total_frames)

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index

        results['frame_inds'] = frame_inds.astype(np.int)
        results['num_wins'] = self.num_wins
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_wins={self.num_wins}, '
                    f'clip_window_alpha={self.alpha}')
        return repr_str

@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, downsample_ratio = 1, **kwargs):
        self.downsample_ratio = downsample_ratio

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        frames = []
        try:
            cap = cv2.VideoCapture(results['filename'])
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if isinstance(frame, np.ndarray):
                    count = count + 1
                    if count % self.downsample_ratio != 0:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    break
            cap.release()
        except Exception as e:
            raise Exception('Can\'t load video {}\n{}'.format(results['filename'], e.message))

        results['video_reader'] = np.array(frames)
        results['total_frames'] = len(frames)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(down_size = {self.downsample_ratio})')
        return repr_str


@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".
    """

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # Generate frame index mapping in order
        frame_dict = {
            idx: container[idx]
            for idx in np.unique(frame_inds)
        }

        imgs = [frame_dict[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

@PIPELINES.register_module()
class RawFrameDecode:
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            filepath = osp.join(directory, filename_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            # Get frame with channel order RGB directly.
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            imgs.append(cur_frame)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str
