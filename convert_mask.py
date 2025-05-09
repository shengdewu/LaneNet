import os
import json
import cv2
import numpy as np


def convert_mask(in_path, out_path):
    """
    :param in_path: 使用 x-anylabel 标注的折线数据 (jpg, json)
    :param out_path:
    :return:
    """
    json_names = [name for name in os.listdir(in_path) if name.endswith('json')]
    for name in json_names:
        with open(f'{in_path}/{name}', mode='r') as f:
            anns = json.load(f)

        arr = name.split('.')
        img = cv2.imread(f'{in_path}/{arr[0]}.jpg', cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(img)
        for shape in anns['shapes']:
            label = int(shape['label'])
            pts = np.asarray(shape['points'], dtype=np.int32)
            """
            不要用 LINE_AA, 他会使用高斯模糊， 当label超过1时，边缘的值会被模糊 不等于label
            """
            cv2.polylines(mask, [pts], isClosed=False, color=label, thickness=4, lineType=cv2.LINE_8)

        cv2.imwrite(f'{out_path}/{arr[0]}.png', mask)

    return


if __name__ == '__main__':
    in_root = '/home/thinkbook/workspace/datasets/pidai'
    out_root = '/home/thinkbook/workspace/datasets/pidai'
    convert_mask(in_root, out_root)

