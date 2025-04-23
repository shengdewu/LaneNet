_base_ = [
    './base/dataset/lane.py',
    './base/model/lane.py',
    './base/schedule/sgd_cosine.py',
]

max_iter = 100
learning_rate = 4e-4

dataloader = dict(
    num_workers=4,
)

trainer = dict(
    device='cuda',
    enable_epoch_method=True
)

solver = dict(
    train_per_batch=16,
    test_per_batch=8,
    max_iter=max_iter,
    checkpoint_period=1,
    generator=dict(
        lr_scheduler=dict(
            enabled=True,
            type='WarmupPolynomialDecay',
            params=dict(
                max_iter=max_iter * 400,
                warmup_factor=0.01,
                warmup_iter=100,
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
)

output_dir = 'checkpoint/lanenet'
