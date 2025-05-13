|名称| 描述                                  | 默认值|
|---|-------------------------------------|---|
|row_h| 图像的高度                               | 288|
|row_anchor| 在 Y 方向取的坐标集合， 减少模型的计算量              | default|
|griding_num| 把 X 方向均匀等分griding_num份              | 100|
|cls_lane| 每条车道线分类个数,必须等于 len(row_anchor),可以省略 | 56 |