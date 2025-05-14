# model settings
cls_num_per_lane = 59
griding_num = 100
num_lanes = 2
pool_channel = 16
cls_channel = 512
ignore_index = 255

model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='RegSegEncoder',
        stem_channels=32,
        stages=[
            [[48, [1], 24, 2, 4], [48, [1], 24, 1, 4]],
            [[120, [1], 24, 2, 4], *[[120, [1], 24, 1, 4]] * 5],
            [
                [336, [1], 24, 2, 4],
                [336, [1], 24, 1, 4],
                [336, [1, 2], 24, 1, 4],
                *[[336, [1, 4], 24, 1, 4]] * 4,
                *[[336, [1, 14], 24, 1, 4]] * 6,
                [384, [1, 14], 24, 1, 4],
            ],
        ],
        out_indices=(0, 1, 2)
    ),
    decoder=dict(
        name='LaneHead',
        in_channels=[48, 120, 384],
        grid_num=griding_num,
        num_lanes=num_lanes,
        cls_num_per_lane=cls_num_per_lane,
        cls_channel=cls_channel,
        pool_channel=pool_channel,
        spp_levels=(1, 2, 4, 8),
        loss_cfg=dict(
            loss=[
                dict(name='SoftmaxFocalLoss',
                     param=dict(gamma=2.0, lambda_weight=10.0, ignore_index=ignore_index)),
                dict(name='SimilarityLoss', param=dict(lambda_weight=2.0), input_name=['logits']),
                dict(name='StraightLoss', param=dict(lambda_weight=2.0), input_name=['logits']),
            ]
        )
    ),
    auxiliary=dict(
        name='LaneHeadAux',
        in_channels=[48, 120, 384],
        aux_channel=128,
        num_lanes=num_lanes,
        loss_cfg=dict(
            loss=[
                dict(name='GeneralizedCELoss',
                     param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index))
            ])
    )
)

trainer = dict(
    name='LaneTrainer',
    weights='',
    enable_epoch_method=False,
    model=dict(
        name='LaneModel',
        generator=model,
    )
)
