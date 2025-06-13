import abc
from typing import Any, Tuple
import time
from tool.py_onnx import PyOnnx
import cv2
import numpy as np


class Engine(abc.ABC):
    """推理接口基类"""

    def __init__(self, onnx_path: str, img_size: Tuple[int]):
        """
        初始化推理接口

        Args:
            onnx_path: 模型onnx文件路径
            img_size: 模型的输入大小 (w, h)
        """

        self.model = PyOnnx(onnx_path)
        self.img_size = img_size
        return

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理, 图像只支持不保持比率缩放

        Args:
            image: 图像 bgr

        Returns:
            处理后的张量
        """
        array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = cv2.resize(array, dsize=self.img_size, interpolation=cv2.INTER_LINEAR)
        array = array.astype(np.float32) / 255

        array = np.stack([array])
        array = np.transpose(array, (0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        return np.ascontiguousarray(array)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        对输入图像进行分割预测

        Args:
            image: 图像数组bgr

        Returns:
            分割掩码（numpy数组）
        """
        array = self.preprocess(image)
        output = self.model(array)[0]

        return self.postprocess(output, image.shape[:2], array.shape[2:])

    @abc.abstractmethod
    def postprocess(self, out: np.ndarray, img_shape: Tuple[int], in_shape: Tuple[int]) -> Any:
        """
        对分割掩码进行后处理

        Args:
            out: 模型输出的结果
            img_shape: 原始图像大小 [h, w]
            in_shape: 模型输入大小 [h, w]
        Returns:
            处理后的结果
        """
        raise NotImplemented('这个方法必须实现')


class PDEngine(Engine):
    def __init__(self, onnx_path='./RegSegEncoder-LaneHead_final.onnx', wh=(640, 384)):
        """
        初始化推理接口

        Args:
            onnx_path: 模型onnx文件路径
            wh: 模型的输入大小 (w, h)
        """
        super(PDEngine, self).__init__(onnx_path, wh)

        self.config = {
            "row_h": 1080,
            "row_anchor": [18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342,
                           360, 378,
                           396, 414, 432, 450, 468, 486, 504, 522, 540, 558, 576, 594, 612, 630, 648, 666, 684, 702,
                           720, 738,
                           756, 774, 792, 810, 828, 846, 864, 882, 900, 918, 936, 954, 972, 990, 1008, 1026, 1044,
                           1062],
            "griding_num": 100
        }

        # self.model_onnx = PyOnnx(onnx_path)  # load onnx model
        self.wh = wh
        return

    @staticmethod
    def softmax(x, axis=None):
        """
        计算输入数组沿指定轴的 Softmax 值

        参数:
        x (np.ndarray): 输入数组
        axis (int, 可选): 计算 Softmax 的轴，默认为 None（处理为一维数组）

        返回:
        np.ndarray: Softmax 结果，形状与输入相同
        """
        # 为了数值稳定性，减去最大值
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

        # 计算指数
        exp_x = np.exp(x)

        # 计算分母
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

        # 返回结果
        return exp_x / sum_exp

    def postprocess(self, out: np.ndarray, img_shape: Tuple[int], in_shape: Tuple[int]) -> Any:
        bs, cls_nums, anchors, nums = out.shape

        # out = torch.flip(out, dims=[2])
        prob = self.softmax(out[:, :-1, ...], axis=1)
        idx = np.arange(1, self.config['griding_num'] + 1)[np.newaxis, ...]
        idx = idx.repeat(bs, 1)
        idx = idx.reshape(bs, -1, 1, 1)
        loc = np.sum(prob * idx, axis=1)
        out = np.argmax(out, axis=1)
        loc[out == self.config['griding_num']] = 0

        h, w = in_shape
        col_sample = np.linspace(0, w - 1, self.config['griding_num'])
        step = col_sample[1] - col_sample[0]

        oh, ow = img_shape
        ws = float(ow) / w
        hs = float(oh) / h
        lines = self.to_lines(loc[0], (h, w), step, w)
        return [[(int(x * ws), int(y * hs)) for x, y in line] for line in lines]

    def to_lines(self, cls_group, img_shape, col_sample_step, in_width=800):
        """
        :param cls_group:
        :param img_shape: 原始图像 (h, w)
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
                     int(h * (self.config['row_anchor'][k] / self.config['row_h'])) - 1)
                line.append(p)
            lines.append(line)
        return lines


class GOOffTrack:
    def __init__(self, threshold=10):
        self.threshold = threshold
        return

    @staticmethod
    def smooth_coordinates(pts: list, window_size=5):
        """
        使用滑动平均平滑坐标
        :param pts: numpy array，形状 (N, 2)
        :param window_size: 滑动窗口大小（必须是奇数）
        :return: 平滑后的点
        """
        if window_size % 2 == 0:
            window_size += 1  # 保证是奇数

        line = np.array(pts)

        pad_size = window_size // 2
        padded = np.pad(line, ((pad_size, pad_size), (0, 0)), mode='edge')
        smoothed = [0] * line.shape[0]
        for i in range(line.shape[0]):
            smoothed[i] = np.mean(padded[i:i + window_size], axis=0).tolist()
        return smoothed

    @staticmethod
    def polyfit(line: list):
        x = [x for x, y in line]
        y = [y for x, y in line]
        if sum(abs(b - a) for a, b in zip(x[:-1], x[1:])) < 1e-3:
            return None, None

        slope, intercept = np.polyfit(np.array(x), np.array(y), deg=1)
        return slope, intercept

    @staticmethod
    def calculate_angle(k1, k2):
        # 计算tan(theta)
        if k1 is None and k2 is None:
            return 0

        if k1 is None:
            if k2 == 0:
                angle_deg = 90
            else:
                angle_deg = np.degrees(np.arctan(abs(1 / k2)))
        elif k2 is None:
            if k1 == 0:
                angle_deg = 0
            else:
                angle_deg = np.degrees(np.arctan(abs(1 / k1)))
        else:
            tan_theta = abs((k2 - k1) / (1 + k1 * k2))
            # 计算角度（弧度转角度）
            angle_rad = np.arctan(tan_theta)
            angle_deg = np.degrees(angle_rad)
        return angle_deg

    def __call__(self, line1, line2):
        line1 = self.smooth_coordinates(line1)
        slope1, intercept1 = self.polyfit(line1)
        slope2, intercept2 = self.polyfit(line2)

        angle = self.calculate_angle(slope1, slope2)
        return angle, line1


if __name__ == '__main__':
    model_onnx = PDEngine('./RegSegEncoder-LaneHead_final-128.onnx')  # load onnx model
    img_bgr = cv2.imread('91.jpg', cv2.IMREAD_COLOR)

    start = time.time()
    lines = model_onnx(img_bgr)
    print(time.time() - start)

    guide1 = [(700, 357), (270, 1057)]
    guide2 = [(1200, 374), (1600, 1057)]

    go_off = GOOffTrack(10)
    slope1, line1 = go_off(lines[0], guide1)
    slope2, line2 = go_off(lines[1], guide2)
    slope3, line3 = go_off(guide1, guide2)

    for line in lines:
        for pt in line:
            cv2.circle(img_bgr, pt, 5, (255, 0, 0), -1)

    for line in [line1, line2]:
        line = [(int(x), int(y)) for x, y in line]
        for pt in line:
            cv2.circle(img_bgr, pt, 5, (0, 0, 255), -1)

    cv2.line(img_bgr, guide1[0], guide1[1], (0, 255, 0))
    cv2.line(img_bgr, guide2[0], guide2[1], (0, 255, 0))

    pt1 = np.mean(guide1, axis=0).astype(int).tolist()
    pt2 = np.mean(guide2, axis=0).astype(int).tolist()
    cv2.putText(img_bgr, f'{round(slope1, 3)}', pt1, 2, 2, (255, 255, 0))
    cv2.putText(img_bgr, f'{round(slope2, 3)}', pt2, 2, 2, (255, 255, 0))

    cv2.imwrite('lane.jpg', img_bgr)
