from tool.py_onnx import PyOnnx
import cv2
import numpy as np


class PDEngine:
    def __init__(self, onnx_path='./RegSegEncoder-LaneHead_final.onnx', wh=(640, 384)):
        """
        config 匹配训练
        """
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

        self.model_onnx = PyOnnx(onnx_path)  # load onnx model
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

    def to_pts(self, cls_group, img_shape, col_sample_step, in_width=800):
        """
        :param cls_group:
        :param img_shape: 原始图像 (h, w)
        :param col_sample_step:
        :param in_width:  网络的输入宽
        :return:
        """

        h, w = img_shape
        anchors, nums = cls_group.shape
        pts = list()
        for i in range(nums):
            if np.sum(cls_group[:, i] != 0) < 2:
                continue
            for k in range(anchors):
                if cls_group[k, i] <= 0:
                    continue

                p = (int(cls_group[k, i] * col_sample_step * w / in_width) - 1,
                     int(h * (self.config['row_anchor'][k] / self.config['row_h'])) - 1)
                pts.append(p)
        return pts

    def __call__(self, in_array: np.ndarray):
        img_rgb = cv2.cvtColor(in_array, cv2.COLOR_BGR2RGB)
        oh, ow, oc = img_rgb.shape

        img_rgb = cv2.resize(img_rgb, dsize=self.wh, interpolation=cv2.INTER_LINEAR)

        h, w, c = img_rgb.shape

        in_np = np.transpose(np.stack([img_rgb.astype(np.float32) / 255]), (0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        in_np = np.ascontiguousarray(in_np)
        logits = self.model_onnx(in_np)[0]
        print(logits.shape)

        bs, cls_nums, anchors, nums = logits.shape

        # logits = torch.flip(logits, dims=[2])
        prob = self.softmax(logits[:, :-1, ...], axis=1)
        idx = np.arange(1, self.config['griding_num'] + 1)[np.newaxis, ...]
        idx = idx.repeat(bs, 1)
        idx = idx.reshape(bs, -1, 1, 1)
        loc = np.sum(prob * idx, axis=1)
        logits = np.argmax(logits, axis=1)
        loc[logits == self.config['griding_num']] = 0

        col_sample = np.linspace(0, w - 1, self.config['griding_num'])
        step = col_sample[1] - col_sample[0]

        ws = float(ow) / w
        hs = float(oh) / h
        pts = self.to_pts(loc[0], (h, w), step, w)
        return [(int(x * ws), int(y * hs)) for x, y in pts]


if __name__ == '__main__':
    model_onnx = PDEngine('./RegSegEncoder-LaneHead_final.onnx')  # load onnx model
    img_bgr = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

    pts = model_onnx(img_bgr)
    for pt in pts:
        cv2.circle(img_bgr, pt, 5, (255, 0, 0), -1)

    cv2.imwrite('lane1.jpg', img_bgr)
