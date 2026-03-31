import os
import json
import random
import math
import tqdm
import cv2
import numpy as np

COLORS = [
    [0, 0, 139],
    [34, 139, 34],
    [230, 216, 173],
    [0, 255, 255],
    [250, 230, 230],
    [128, 0, 0]
]


def polyfit(line: list):
    x = [x for x, y in line]
    y = [y for x, y in line]
    if sum(abs(b - a) for a, b in zip(x[:-1], x[1:])) < 1e-3:
        return None, None

    slope, intercept = np.polyfit(np.array(x), np.array(y), deg=1)
    return slope, intercept


def rotate_image(image: np.ndarray, clockwise=True):
    """
    旋转图像90度并同步转换点坐标

    Args:
        image: 原始图像
        clockwise: 是否顺时针旋转（默认True）

    Returns:
        rotated_img: 旋转后的图像
        rotated_points: 旋转后的点坐标，shape=(n, 2)
    """
    # 旋转图像90度
    if clockwise:
        # 顺时针旋转90度，旋转后尺寸为(w, h)
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        # 逆时针旋转90度
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rotated_img


def rotate_points(shape, points: list[tuple], clockwise=True):
    """
    旋转图像90度并同步转换点坐标

    参数:
        shape: 原始图像shape
        points: 点 ，每个点为(x, y)，shape=(n, 2)
        clockwise: 是否顺时针旋转（默认True）

    返回:
        rotated_img: 旋转后的图像
        rotated_points: 旋转后的点坐标，shape=(n, 2)
    """
    h, w = shape[:2]

    # 转换关键点坐标
    rotated_points = []
    for (x, y) in points:
        if clockwise:
            # 顺时针旋转90度坐标变换公式: (x, y) → (y, w - 1 - x)
            new_x = h - 1 - y
            new_y = x
        else:
            # 逆时针旋转90度坐标变换公式: (x, y) → (h - 1 - y, x)
            new_x = y
            new_y = w - 1 - x
        rotated_points.append([new_x, new_y])

    return rotated_points


def create_mask(in_path, out_path, out_weight):
    """
    生成边缘mask

    Args
        in_path: 使用 x-anylabel 标注的折线数据 (jpg, json) 标签从 1 开始
        out_path:
        out_weight: 掩码测试路径
    """
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_weight, exist_ok=True)

    def decay(p0, p1, max_dis):
        dis = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
        return 1 - min(1.0, math.exp(dis / max_dis * 1.4) - 1)

    def decay_y(y0, y1):
        return math.exp(-2 * (1 - y0 / y1))

    json_names = [name for name in os.listdir(in_path) if name.endswith('json')]
    for name in tqdm.tqdm(json_names):

        with open(f'{in_path}/{name}', mode='r') as f:
            anns = json.load(f)

        tmp = dict()
        for shape in anns['shapes']:
            label = int(shape['label'])
            if tmp.get(label, None) is None:
                tmp[label] = list()
            tmp[label].extend(shape['points'])

        arr = name.split('.')
        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_GRAYSCALE)

        shapes = dict()
        for k, pts in tmp.items():
            shapes[k] = sorted(pts, key=lambda xy: xy[1], reverse=False)

        del tmp

        mask_name = f'{out_path}/{arr[0]}-line.png'
        h, w = img.shape
        # max_len = max(h, w)  # (h * h + w * w) ** 0.5
        mask = np.zeros_like(img)
        offset_base = w // 200  # 图片被均匀分成100份, 参照训练的配置文件
        # labels = list(shapes.keys())
        for label, points in shapes.items():
            """
            皮带边缘标签 依次是 1, 2, 3, 4, ... 
            1, 2 是一个皮带, 3, 4 是一个皮带
            扩张的时候 往皮带内部扩张  
            通过两个皮带边缘的距离来确定offset的大小,防止在远处两个皮带边缘沾连
            """

            start = points[-1]
            if label % 2 == 0:  # 是偶数 则往右边扩张
                offset = -offset_base
            else:  # 是奇数则往左阔边扩张
                offset = offset_base

            in_pts = [[pt[0] + offset * decay_y(pt[1], start[1]), pt[1]] for pt in points]
            in_pts = in_pts[::-1]
            pts = points + in_pts
            pts = np.asarray(pts, dtype=np.int32)

            """
            不要用 LINE_AA, 他会使用高斯模糊， 当label超过1时，边缘的值会被模糊 不等于label
            """
            # cv2.polylines(mask, [pts], isClosed=False, color=label, thickness=4, lineType=cv2.LINE_8)
            cv2.fillPoly(mask, [pts], color=label, lineType=cv2.LINE_8)

        if flip == 'vflip':
            direct = dict(vflip=True)
            mask = rotate_image(mask, clockwise=not direct[flip])

        cv2.imwrite(mask_name, mask)

        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_COLOR)
        for label in np.unique(mask):
            if label == 0:
                continue
            b, g, r = COLORS[label]
            img[:, :, 0] = np.where(mask == label, b, img[:, :, 0])
            img[:, :, 1] = np.where(mask == label, g, img[:, :, 1])
            img[:, :, 2] = np.where(mask == label, r, img[:, :, 2])

        cv2.imwrite(f'{out_weight}/{arr[0]}_weight.jpg', img)

    return


def create_anchor(in_path, cls_lane=56):
    json_names = [name for name in os.listdir(in_path) if name.endswith('json')]
    row_anchor = dict()
    for name in json_names:
        with open(f'{in_path}/{name}', mode='r') as f:
            anns = json.load(f)

        height = anns['imageHeight']
        if row_anchor.get(height, None) is None:
            row_anchor[height] = list()

        for shape in anns['shapes']:
            pts = [pt[1] for pt in shape['points']]
            pts = sorted(pts, reverse=False)
            row_anchor[height].append(pts[0])

    for key, vals in row_anchor.items():
        vals = sorted(vals, reverse=False)
        print(f'{key} ({min(vals)},{max(vals)}) {vals[len(vals) // 2]} {sum(vals) / len(vals)}')

    step = 1080 // cls_lane
    print([i for i in range(step, 1080, step)])


if __name__ == '__main__':
    in_root = '/mnt/sda/datasets/边缘数据集合/part3'
    out_root = '/mnt/sda/datasets/边缘数据集合/part3'
    create_mask(in_root, out_root, f'{out_root}-weight')

    # create_anchor(in_root, 60)

    in_root = '/mnt/sda/datasets/边缘数据集合'
    names = list()
    for pname in ['part0', 'part2', 'part3']:
        names.extend([f'{pname}/{name}' for name in os.listdir(f'{in_root}/{pname}') if name.endswith('png')])

    random.shuffle(names)
    idx = int(len(names) * 0.9)
    with open(f'{in_root}/train_part023.txt', mode='w') as f:
        for name in names[: idx]:
            arr = name.split('.')
            f.write(f'{arr[0]}.jpg {arr[0]}.png\n')

    with open(f'{in_root}/test_part023.txt', mode='w') as f:
        for name in names[idx:]:
            arr = name.split('.')
            f.write(f'{arr[0]}.jpg {arr[0]}.png\n')
