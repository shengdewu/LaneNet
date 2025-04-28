# model settings
num_lanes = 4
grid_num = 100
ignore_index = 255

model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[2, 3, 4],
        stem_sharp=False
    ),
    decoder=dict(
        name='LaneHead',
        in_channels=[128, 256, 512],
        grid_num=grid_num,
        num_lanes=num_lanes,
        cls_num_per_lane=56,
        cls_channel=1024,
        pool_channel=8,
        spp_levels=(1, 2, 4, 6),
        loss_cfg=dict(
            loss=[
                dict(name='SoftmaxFocalLoss',
                     param=dict(gamma=2.0, lambda_weight=1.0, ignore_index=ignore_index)),
                dict(name='SimilarityLoss', param=dict(lambda_weight=1.0), input_name=['logits']),
                dict(name='StraightLoss', param=dict(lambda_weight=1.0), input_name=['logits']),
            ]
        )
    ),
    auxiliary=dict(
        name='LaneHeadAux',
        in_channels=[128, 256, 512],
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
