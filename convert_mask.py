import os
import json
import random
import shutil

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


def create_mask(in_path, out_path):
    """
    :param in_path: 使用 x-anylabel 标注的折线数据 (jpg, json) 标签从 1 开始
    :param out_path:
    :return:
    """
    os.makedirs(out_path, exist_ok=True)

    json_names = [name for name in os.listdir(in_path) if name.endswith('json')]
    for name in json_names:
        with open(f'{in_path}/{name}', mode='r') as f:
            anns = json.load(f)

        tmp = dict()
        for shape in anns['shapes']:
            label = int(shape['label'])
            if tmp.get(label, None) is None:
                tmp[label] = list()
            tmp[label].extend(shape['points'])

        shapes = dict()
        for k, pts in tmp.items():
            shapes[k] = sorted(pts, key=lambda xy: xy[1], reverse=False)

        del tmp
        arr = name.split('.')
        mask_name = f'{out_path}/{arr[0]}.png'

        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(img)
        offset_base = 20
        for label, points in shapes.items():
            """
            皮带边缘标签 依次是 1, 2, 3, 4, ... 
            1, 2 是一个皮带, 3, 4 是一个皮带
            扩张的时候 往皮带内部扩张  
            通过两个皮带边缘的距离来确定offset的大小,防止在远处两个皮带边缘沾连
            """
            if label % 2 == 0:  # 是偶数 则往右边扩张
                offset = -offset_base
            else:  # 是奇数则往左阔边扩张
                offset = offset_base

            if label % 2 == 1:  # 是奇数 则和它同属一条皮带的label = label + 1
                nb_label = label + 1
            else:
                nb_label = label - 1

            nb_points = shapes.get(nb_label, None)
            if nb_points is not None:
                nb_k, nb_b = polyfit(nb_points)

                front_x = points[-1][0]
                back_x = points[0][0]

                if nb_k is None:
                    nb_front_x = shapes[nb_label][-1][0]
                    nb_back_x = shapes[nb_label][0][0]
                else:
                    nb_front_x = (points[-1][1] - nb_b) / nb_k
                    nb_back_x = (points[0][1] - nb_b) / nb_k

                back_offset = abs(back_x - nb_back_x)
                front_offset = abs(front_x - nb_front_x)

                if back_offset <= abs(offset * 5) or front_offset > back_offset * 5:
                    back_y = back_offset / front_offset
                    front_y = 1

                    if front_x != back_x:
                        k = (front_y - back_y) / (front_x - back_x)
                        b = front_y - front_x * k
                    else:
                        k = 0
                        b = 1

                    in_pts = [[pt[0] + offset * (k * pt[0] + b), pt[1]] for pt in points]
                    in_pts = in_pts[::-1]
                    pts = points + in_pts
                else:
                    in_pts = [[pt[0] + offset, pt[1]] for pt in points]
                    in_pts = in_pts[::-1]
                    pts = points + in_pts

                pts = np.asarray(pts, dtype=np.int32)
            else:
                in_pts = [[pt[0] + offset, pt[1]] for pt in points]
                in_pts = in_pts[::-1]
                pts = points + in_pts

                pts = np.asarray(pts, dtype=np.int32)
            """
            不要用 LINE_AA, 他会使用高斯模糊， 当label超过1时，边缘的值会被模糊 不等于label
            """
            # cv2.polylines(mask, [pts], isClosed=False, color=label, thickness=4, lineType=cv2.LINE_8)
            cv2.fillPoly(mask, [pts], color=label, lineType=cv2.LINE_8)

        cv2.imwrite(mask_name, mask)

        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_COLOR)
        for label in np.unique(mask):
            if label == 0:
                continue
            b, g, r = COLORS[label]
            img[:, :, 0] = np.where(mask == label, b, img[:, :, 0])
            img[:, :, 1] = np.where(mask == label, g, img[:, :, 1])
            img[:, :, 2] = np.where(mask == label, r, img[:, :, 2])

        cv2.imwrite(f'{out_path}/{arr[0]}_weight.jpg', img)

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
    in_root = '/mnt/sda/datasets/皮带跑偏数据集合/part3'
    out_root = '/mnt/sda/datasets/皮带跑偏数据集合/part3'
    create_mask(in_root, out_root)

    # create_anchor(in_root, 60)

    in_root = '/mnt/sda/datasets/皮带跑偏数据集合'
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
