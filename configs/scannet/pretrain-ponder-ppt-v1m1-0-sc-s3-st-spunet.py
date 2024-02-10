_base_ = ["../_base_/default_runtime.py"]

num_gpu = 8
limit_num_coord = 2000000

# misc custom setting
batch_size = 8 * num_gpu  # bs: total bs in all gpus
num_worker = 16 * num_gpu

mix_prob = 0.0
empty_cache = True
enable_amp = True
evaluate = False
find_unused_parameters = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="PonderIndoor-v2",
    backbone=dict(
        type="SpUNet-v1m3",
        in_channels=6,
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        zero_init=False,
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=True,
    ),
    projection=dict(
        type="UNet3D-v1m2",
        in_channels=96,
        out_channels=128,
    ),
    renderer=dict(
        type="NeuSModel",
        field=dict(
            type="SDFField",
            sdf_decoder=dict(
                in_dim=128,
                out_dim=65,  # 64 + 1
                hidden_size=128,
                n_blocks=1,
            ),
            rgb_decoder=dict(
                in_dim=198,  # 128 + 64 + 3 + 3
                out_dim=3,
                hidden_size=128,
                n_blocks=0,
            ),
            semantic_decoder=dict(
                in_dim=195,  # 128 + 64 + 3, no directions
                out_dim=512,
                hidden_size=128,
                n_blocks=0,
            ),
            beta_init=0.3,
            use_gradient=True,
            volume_type="default",
            padding_mode="zeros",
            share_volume=True,
            norm_pts=True,
            norm_padding=0.1,
        ),
        collider=dict(
            type="AABBBoxCollider",
            near_plane=0.01,
            bbox=[-0.55, -0.55, -0.55, 0.55, 0.55, 0.55],
        ),
        sampler=dict(
            type="NeuSSampler",
            initial_sampler="UniformSampler",
            num_samples=96,
            num_samples_importance=36,
            num_upsample_steps=1,
            train_stratified=True,
            single_jitter=False,
        ),
        loss=dict(
            sensor_depth_truncation=0.05,
            temperature=0.01,
            weights=dict(
                eikonal_loss=0.01,
                free_space_loss=1.0,
                sdf_loss=10.0,
                depth_loss=1.0,
                rgb_loss=10.0,
                semantic_loss=0.1,
            ),
        ),
    ),
    # mask=dict(ratio=0.8, size=8, channel=6),
    mask=None,
    grid_shape=(128, 128, 32),
    grid_size=0.02,
    val_ray_split=10240,
    ray_nsample=256,
    padding=0.1,
    backbone_out_channels=96,
    context_channels=256,
    pool_type="mean",
    share_volume=True,
    render_semantic=True,
    conditions=("Structured3D", "ScanNet", "S3DIS"),
    template=(
        "itap of a [x]",
        "a origami [x]",
        "a rendering of a [x]",
        "a painting of a [x]",
        "a photo of a [x]",
        "a photo of one [x]",
        "a photo of a nice [x]",
        "a photo of a weird [x]",
        "a cropped photo of a [x]",
        "a bad photo of a [x]",
        "a good photo of a [x]",
        "a photo of the large [x]",
        "a photo of the small [x]",
        "a photo of a clean [x]",
        "a photo of a dirty [x]",
        "a bright photo of a [x]",
        "a dark photo of a [x]",
        "a [x] in a living room",
        "a [x] in a bedroom",
        "a [x] in a kitchen",
        "a [x] in a bathroom",
    ),
    clip_model="ViT-B/16",
    class_name=(
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "bookcase",
        "picture",
        "counter",
        "desk",
        "shelves",
        "curtain",
        "dresser",
        "pillow",
        "mirror",
        "ceiling",
        "refrigerator",
        "television",
        "shower curtain",
        "nightstand",
        "toilet",
        "sink",
        "lamp",
        "bathtub",
        "garbagebin",
        "board",
        "beam",
        "column",
        "clutter",
        "other structure",
        "other furniture",
        "other property",
    ),
    valid_index=(
        (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            25,
            26,
            33,
            34,
            35,
        ),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
        (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
    ),
    ppt_loss_weight=0.0,
    ppt_criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 400
eval_epoch = 1
optimizer = dict(
    type="SGD",
    lr=0.0001 * batch_size / 8,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True,
)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# dataset settings
num_cameras = 5
data = dict(
    num_classes=20,
    ignore_index=-1,
    names=(
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ),
    train=dict(
        type="ConcatDataset",
        datasets=[
            # Structured3D
            dict(
                type="Structured3DRGBDDataset",
                split="train",
                data_root="data/structured3d",
                render_semantic=True,
                num_cameras=num_cameras,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.8,
                        dropout_application_ratio=1.0,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1 / 64, 1 / 64],
                        axis="x",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1 / 64, 1 / 64],
                        axis="y",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomScale",
                        scale=[0.9, 1.1],
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomFlip",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(
                        type="CenterShift",
                        apply_z=False,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "Structured3D"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "condition",
                            "rgb",
                            "depth",
                            "depth_scale",
                        ),
                        stack_keys=(
                            "intrinsic",
                            "extrinsic",
                            "rgb",
                            "depth",
                            "semantic",
                        ),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=4,  # sampling weight
            ),
            # ScanNet
            dict(
                type="ScanNetRGBDDataset",
                split="train",
                data_root="data/scannet",
                rgbd_root="data/scannet/rgbd",
                render_semantic=True,
                num_cameras=num_cameras,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.8,
                        dropout_application_ratio=1.0,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1 / 64, 1 / 64],
                        axis="x",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1 / 64, 1 / 64],
                        axis="y",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomScale",
                        scale=[0.9, 1.1],
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomFlip",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(
                        type="CenterShift",
                        apply_z=False,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "ScanNet"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "condition",
                            "rgb",
                            "depth",
                            "depth_scale",
                        ),
                        stack_keys=(
                            "intrinsic",
                            "extrinsic",
                            "rgb",
                            "depth",
                            "semantic",
                        ),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=2,  # sampling weight
            ),
            # S3DIS
            dict(
                type="S3DISRGBDDataset",
                split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
                data_root="data/s3dis",
                render_semantic=True,
                num_cameras=num_cameras,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.8,
                        dropout_application_ratio=1.0,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1 / 64, 1 / 64],
                        axis="x",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1 / 64, 1 / 64],
                        axis="y",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="RandomScale",
                        scale=[0.9, 1.1],
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(
                        type="RandomFlip",
                        p=0.5,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
                    dict(
                        type="CenterShift",
                        apply_z=False,
                        keys=[
                            "extrinsic",
                        ],
                    ),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "S3DIS"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "condition",
                            "rgb",
                            "depth",
                            "depth_scale",
                        ),
                        stack_keys=(
                            "intrinsic",
                            "extrinsic",
                            "rgb",
                            "depth",
                            "semantic",
                        ),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=1,  # sampling weight
            ),
        ],
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
