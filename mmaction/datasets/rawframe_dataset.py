import copy
import os
import os.path as osp
import random

import torch

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class RawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 0
        some/directory-2 0.33
        some/directory-3 0.33
        some/directory-4 0.66
        some/directory-5 0.66
        some/directory-6 1


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='frame_det_00_{:06}.bmp',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 duplicate_times = [1, 1, 1, 1]):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.duplicate_times = duplicate_times
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)

    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                frame_dir, label = line_split
                if type(eval(label))==float:
                    # Engagmentwild dataset
                    video_info['label'] = float(label)
                else:
                    # DAiSEE dataset
                    video_info['label'] = [0.0, 0.33, 0.66, 1.0][int(label)]
                    # video_info['label'] = int(label)
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir

                # walk video path to find the number of frames
                frame_path_suffix = self.filename_tmpl.split('.')[-1]
                total_frames = 0
                for frame_path in os.listdir(frame_dir):
                    if frame_path.endswith(frame_path_suffix):
                        total_frames = total_frames + 1
                video_info['total_frames'] = total_frames

                # Duplicate video info to generate multiple instances
                label_index = [0.0, 0.33, 0.66, 1.0].index(video_info['label'])
                video_duplicate_times = self.duplicate_times[label_index]

                int_part = int(video_duplicate_times)
                for i in range(int_part):
                    video_info_copy = video_info.copy()
                    video_info_copy['clip_start_idx'] = i
                    video_infos.append(video_info_copy)
                decimal_part = video_duplicate_times - int_part
                if decimal_part > 0:
                    if random.uniform(0, 1) < (video_duplicate_times - int(video_duplicate_times)):
                        video_info_copy = video_info.copy()
                        video_info_copy['clip_start_idx'] = int(video_duplicate_times) + 1
                        video_infos.append(video_info_copy)

        # Shuffle video sequence to be trained
        if not self.test_mode:
            random.shuffle(video_infos)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results = self.pipeline(results)

        return results

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results = self.pipeline(results)

        return results
