import os
import json
import shutil

import cv2
import numpy as np


def convert_mask(in_path, out_path):
    """
    :param in_path: 使用 x-anylabel 标注的折线数据 (jpg, json)
    :param out_path:
    :return:
    """
    os.makedirs(out_path, exist_ok=True)

    json_names = [name for name in os.listdir(in_path) if name.endswith('json')]
    for name in json_names:
        with open(f'{in_path}/{name}', mode='r') as f:
            anns = json.load(f)

        if not os.path.exists(f'{out_path}/{name}'):
            shutil.copy2(f'{in_path}/{name}', f'{out_path}/{name}')

        arr = name.split('.')
        mask_name = f'{out_path}/{arr[0]}.png'
        if os.path.exists(mask_name):
            continue

        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_GRAYSCALE)

        if not os.path.exists(f'{out_path}/{arr[0]}.jpg'):
            shutil.copy2(f'{in_path}/{arr[0]}.jpg', f'{out_path}/{arr[0]}.jpg')

        mask = np.zeros_like(img)
        for shape in anns['shapes']:
            label = int(shape['label'])
            offset = 20 if label == 1 else -20
            in_pts = [[pt[0] + offset, pt[1]] for pt in shape['points']]
            in_pts = in_pts[::-1]
            pts = shape['points'] + in_pts

            pts = np.asarray(pts, dtype=np.int32)
            """
            不要用 LINE_AA, 他会使用高斯模糊， 当label超过1时，边缘的值会被模糊 不等于label
            """
            # cv2.polylines(mask, [pts], isClosed=False, color=label, thickness=4, lineType=cv2.LINE_8)
            cv2.fillPoly(mask, [pts], color=label, lineType=cv2.LINE_8)

        cv2.imwrite(mask_name, mask)

        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_COLOR)
        weight = np.zeros_like(img)
        weight[:, :, 0] = mask * 100
        overlapping = cv2.addWeighted(img, 0.65, weight, 0.35, 0)
        weight[:, :, 1] = weight[:, :, 0]
        weight[:, :, 2] = weight[:, :, 0]
        overlapping = np.where(weight > 0, overlapping, img)
        cv2.imwrite(f'{out_path}/{arr[0]}_weight.jpg', overlapping)

    return


if __name__ == '__main__':
    in_root = '/home/thinkbook/workspace/datasets/pidai-src'
    out_root = '/home/thinkbook/workspace/datasets/pidai-2'
    convert_mask(in_root, out_root)

