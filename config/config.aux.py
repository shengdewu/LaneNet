import json

lane_config = '/home/thinkbook/workspace/LaneNet/config/base/dataset/pidai.json'
with open(lane_config, mode='r') as f:
    cfg = json.load(f)

cls_num_per_lane = len(cfg['row_anchor'])
griding_num = cfg['griding_num']
num_lanes = 4
pool_channel = 16
cls_channel = 128

batch_size = 8
num_workers = 4
img_size = (384, 640)
max_iter = 12000
warmup_iter = 100
checkpoint_period = 500
is_poly = True
steps = [2000, 4000, 6000, 7500]  # is multi
enable_epoch_method = False
learning_rate = 4e-4
ignore_index = 255
use_all_pts = True

img_root = '/mnt/sda/datasets/皮带跑偏数据集合'
output_dir = f'/mnt/sda/train.output/lanenet/lanenet-4lane-aux-d023-12000-dice-use-{use_all_pts}'

t_transformer = [
    # dict(
    #     name='RandomFlip',
    #     direction=['vertical'],
    #     p=0.4,
    # ),
    dict(
        name='RandomAffine',
        rotate_degree_range=[degree for degree in range(-6, 6, 1)],
        rotate_range=False,
        border_val=0,
        p=0.1,
    ),
    dict(name='RandomCrop',
         min_crop_ratio=0.05,
         max_crop_ratio=0.25,
         crop_step=0.05,
         p=0.8),
    dict(name='Resize',
         interpolation='INTER_LINEAR',
         target_size=img_size,
         keep_ratio=False,
         is_padding=False),
    dict(
        name='RandomCompress',
        quality_lower=75,
        quality_upper=85,
        quality_step=5,
        p=0.5),
    dict(name='RandomColorJitter',
         brightness_limit=[0.5, 1.2],
         brightness_p=0.7,
         contrast_limit=[0.7, 1.3],
         contrast_p=0.6,
         saturation_limit=[0.7, 1.4],
         saturation_p=0.3,
         hue_limit=0.1,
         hue_p=0.2,
         blur_limit=[3, 5],
         sigma_limit=0,
         blur_p=0.2,
         gamma_limit=[0.3, 2.0],
         gamma_p=0.2,
         clahe_limit=4,
         clahe_p=0.),
    dict(name='Normalize',
         mean=(0, 0, 0),
         std=(255, 255, 255))
]

v_transformer = [
    dict(name='Resize',
         interpolation='INTER_LINEAR',
         target_size=img_size,
         keep_ratio=False),
    dict(
        name='RandomCompress',
        quality_lower=75,
        quality_upper=99,
        quality_step=5,
        p=0.5),
    dict(name='Normalize',
         mean=(0, 0, 0),
         std=(255, 255, 255))
]

dataloader = dict(
    num_workers=num_workers,
    train_data_set=dict(
        name='LaneClsDataset',
        path=img_root,
        lane_config=lane_config,
        num_lanes=num_lanes,
        file_names=['train_part0.txt', 'train_part1.txt', 'train_part2.txt', 'train_part3.txt', 'train_part4.txt'],
        transformers=t_transformer,
        aux_is_seg=False
    ),
    val_data_set=dict(
        name='LaneClsDataset',
        path=img_root,
        lane_config=lane_config,
        num_lanes=num_lanes,
        file_names=['test_part0.txt', 'test_part1.txt', 'test_part2.txt', 'test_part3.txt', 'test_part4.txt'],
        transformers=v_transformer,
        aux_is_seg=False
    )
)

model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='RegSegEncoder',
        stem_channels=32,
        stages=[
            # out_channels, dilations, group_width, stride, se_ratio
            [[48, [1], 16, 2, 4]],
            [[96, [1], 16, 2, 4], *[[96, [1], 16, 1, 4]] * 2],
            [
                [128, [1], 16, 2, 4],
                [128, [1], 16, 1, 4],
                [128, [1, 2], 16, 1, 4],
                *[[128, [1, 4], 16, 1, 4]] * 4,
                *[[128, [1, 14], 16, 1, 4]] * 6,
                [160, [1, 14], 16, 1, 4],
            ],
        ],
        out_indices=(0, 1, 2)
    ),
    auxiliary=dict(
        name='LaneHead',
        in_channels=[48, 96, 160],
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
                dict(name='StraightLoss', param=dict(lambda_weight=2.0, use_all_pts=use_all_pts), input_name=['logits']),
            ]
        )
    ),
    decoder=dict(
        name='LaneHeadAux',
        in_channels=[48, 96, 160],
        aux_channel=128,
        num_lanes=num_lanes,
        loss_cfg=dict(
            loss=[
                dict(name='GeneralizedCELoss',
                     param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
                dict(name='GeneralizedDiceLoss',
                     param=dict(lambda_weight=1.0, apply_softmax=False, ignore_index=ignore_index)),
                # dict(name='SSIMLoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
                # dict(name='IOULoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
            ])
    )
)

trainer = dict(
    name='LaneTrainer',
    weights='',
    device='cuda',
    enable_epoch_method=enable_epoch_method,
    model=dict(
        name='LaneModel',
        generator=model,
    )
)

multi_sgd = dict(
    lr_scheduler=dict(
        enabled=True,
        type='LRMultiplierScheduler',
        params=dict(
            lr_scheduler_param=dict(
                name='WarmupCosineLR',
                gamma=0.1,
                steps=steps,
            ),
            warmup_factor=0.01,
            warmup_iter=warmup_iter,
            max_iter=max_iter,
        )
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            momentum=0.9,
            lr=learning_rate,
            weight_decay=5E-4,
        )
    ),
    clip_gradients=dict(
        enabled=False,
    ),
)

poly_adam = dict(
    lr_scheduler=dict(
        enabled=True,
        type='WarmupPolynomialDecay',
        params=dict(
            max_iter=max_iter,
            warmup_factor=0.01,
            warmup_iter=warmup_iter,
            power=0.9,
            end_lr=0.00001,
            simple=True
        )
    ),
    optimizer=dict(
        type='Adam',
        params=dict(
            lr=learning_rate,
            weight_decay=5E-4,
        )
    )
)

solver = dict(
    train_per_batch=batch_size,
    test_per_batch=4,
    max_iter=max_iter,
    max_keep=20,
    checkpoint_period=checkpoint_period,
    generator=poly_adam if is_poly else multi_sgd
)
