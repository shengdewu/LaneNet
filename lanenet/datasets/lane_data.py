from typing import List, Union
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import engine.transforms.functional as F
from engine.data.build import BUILD_DATASET_REGISTRY
from engine.transforms import TransformCompose
import os
import json

__all__ = [
    'LaneClsDataset'
]


@BUILD_DATASET_REGISTRY.register()
class LaneClsDataset(Dataset):

    def __init__(self, path,
                 file_names: Union[str, List],
                 lane_config,
                 transformers: List = None,
                 num_lanes=4,
                 aux_is_seg=True,
                 extend_line=False
                 ):
        super(LaneClsDataset, self).__init__()

        with open(lane_config, mode='r') as f:
            cfg = json.load(f)
        self.griding_num = cfg['griding_num']
        self.row_anchor = cfg['row_anchor']
        self.row_h = cfg['row_h']

        self.transforms = TransformCompose(transformers)
        self.path = path
        self.num_lanes = num_lanes
        self.aux_is_seg = aux_is_seg
        self.extend_line = extend_line

        if isinstance(file_names, str):
            file_names = [file_names]

        self.file_list = list()

        for file_name in file_names:
            with open(f'{path}/{file_name}', 'r') as f:
                file_list = f.readlines()

            for item in file_list:
                l_info = item.strip().split(',')
                img_name, label_name = l_info[0], l_info[1]
                if os.path.exists(f'{self.path}/{img_name}') and os.path.exists(f'{self.path}/{label_name}'):
                    self.file_list.append((img_name, label_name))

        if len(self.file_list) == 0:
            raise FileNotFoundError(f'{path}/{file_names}')
        return

    def __getitem__(self, index):
        img_name, label_name = self.file_list[index]

        label = cv2.imread(f'{self.path}/{label_name}', cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(cv2.imread(f'{self.path}/{img_name}', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        unique = np.unique(label)
        for u in unique:
            if u == 0:
                continue
            if u > self.num_lanes:
                label = np.where(label == u, 0, label)

        result = dict(
            img=img,
            label=label,
            img_fields=['img', 'label'],
            color_fields=['img'],
            interpolation=dict(label='INTER_NEAREST', img='INTER_LINEAR'),
        )

        result = self.transforms(result)

        label = result['label']
        img = result['img']

        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors
        h, w, c = img.shape
        cls_label, step = self._grid_pts(lane_pts, self.griding_num, w)

        # make the coordinates to classification label
        data = dict()
        if self.aux_is_seg:
            data['label'] = torch.from_numpy(cls_label).long()
            data['aux_label'] = F.to_tensor(label).long()
        else:
            data['aux_label'] = torch.from_numpy(cls_label).long()
            data['label'] = F.to_tensor(label).long()

        data['img'] = F.to_tensor(result['img'])
        data['col_sample_step'] = step
        data['grid_num'] = self.griding_num

        return data

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def _grid_pts(pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        step = col_sample[1] - col_sample[0]
        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // step) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int), step

    @staticmethod
    def find_start_pos(row_sample, start_line):
        l, r = 0, len(row_sample) - 1
        while True:
            if r - l == 1:
                return r
            mid = int((l + r) / 2)
            if row_sample[mid] < start_line:
                l = mid
            if row_sample[mid] > start_line:
                r = mid
            if row_sample[mid] == start_line:
                return mid

    def _get_index(self, label):
        h, w = label.shape[:2]

        if h != self.row_h:
            scale = lambda x: int((x * 1.0 / self.row_h) * h)
            sample_tmp = list(map(scale, self.row_anchor))
        else:
            sample_tmp = self.row_anchor

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = label[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                else:
                    pos = np.mean(pos)
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = pos

        if self.extend_line:
            # data augmentation: extend the lane to the boundary of image
            for i in range(self.num_lanes):
                if np.all(all_idx[i, :, 1] == -1):
                    continue
                # if there is no lane

                valid = all_idx[i, :, 1] != -1
                # get all valid lane points' index
                valid_idx = all_idx[i, valid, :]
                # get all valid lane points
                if valid_idx[-1, 0] == all_idx[i, -1, 0]:
                    # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                    # this means this lane has reached the bottom boundary of the image
                    # so we skip
                    continue
                if len(valid_idx) < 6:
                    continue
                # if the lane is too short to extend

                valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
                p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
                start_line = valid_idx_half[-1, 0]
                pos = self.find_start_pos(all_idx[i, :, 0], start_line) + 1

                fitted = np.polyval(p, all_idx[i, pos:, 0])
                fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

                assert np.all(all_idx[i, pos:, 1] == -1)
                all_idx[i, pos:, 1] = fitted
        return all_idx

    def to_lines(self, cls_group, img_shape, col_sample_step, in_width=800):
        """
        :param cls_group:
        :param img_shape: 原始图像
        :param col_sample_step:
        :param in_width:  网络的输入宽
        :return:
        """
        h, w = img_shape
        anchors, nums = cls_group.shape
        lines = list()
        for i in range(nums):
            line = list()
            if np.sum(cls_group[:, i] != 0) < 2:
                continue
            for k in range(anchors):
                if cls_group[k, i] <= 0:
                    continue

                p = (int(cls_group[k, i] * col_sample_step * w / in_width) - 1,
                     int(h * (self.row_anchor[k] / self.row_h)) - 1)
                line.append(p)
            lines.append(line)
        return lines

    def seg_to_lines(self, logit: np.ndarray, img_shape, nc):
        """
        分割结果转点位
        :param logit: h, w 经过了argmax dtype=int64
        :param img_shape: 原始图像h, w
        :param nc: 类别总数
        :return:
        """
        h, w = logit.shape

        ws = float(img_shape[1]) / w
        hs = float(img_shape[0]) / h

        lines = list()
        for i in range(1, nc):
            coord = np.column_stack(np.where(logit == i))
            if coord.shape[0] == 0:
                lines.append([])
                continue
            lines.append([(int(cd[1] * ws), int(cd[0] * hs)) for cd in coord])
        return lines
