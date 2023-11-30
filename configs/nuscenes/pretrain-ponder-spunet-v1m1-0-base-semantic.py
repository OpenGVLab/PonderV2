_base_ = ["../_base_/default_runtime.py"]

num_gpu = 4
# misc custom setting
batch_size = 4 * num_gpu  # bs: total bs in all gpus
num_worker = 8 * num_gpu

mix_prob = 0
empty_cache = True
enable_amp = True
evaluate = False
find_unused_parameters = True

# model settings
model = dict(
    type="PonderOutdoor-v2",
    mask=dict(ratio=0.8, size=8, channel=4),
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=4,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    projection=dict(
        type="SimpleConv3D-v1m1",
        in_channels=96,
        out_channels=32,
    ),
    renderer=dict(
        type="NeuSModel",
        field=dict(
            type="SDFField",
            sdf_decoder=dict(
                in_dim=32,
                out_dim=16 + 1,
                hidden_size=16,
                n_blocks=5,
            ),
            semantic_decoder=dict(
                in_dim=16 + 32 + 3,
                out_dim=512,
                hidden_size=16,
                n_blocks=3,
            ),
            beta_init=0.3,
            use_gradient=True,
            volume_type="default",
            padding_mode="zeros",
            share_volume=True,
        ),
        collider=dict(
            type="AABBBoxCollider",
            near_plane=0.01,
            bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        sampler=dict(
            type="NeuSSampler",
            initial_sampler="UniformSampler",
            num_samples=72,
            num_samples_importance=24,
            num_upsample_steps=1,
            train_stratified=True,
            single_jitter=False,
        ),
        loss=dict(
            sensor_depth_truncation=0.01,
            temperature=0.01,
            weights=dict(
                depth_loss=10.0,
                semantic_loss=0.1,
            ),
        ),
    ),
    scene_bbox=((-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),),
    grid_shape=((180, 180, 5),),
    grid_size=((0.6, 0.6, 1.6),),
    val_ray_split=8192,
    pool_type="mean",
    share_volume=True,
    render_semantic=True,
    conditions=("nuScenes",),
    template="[x]",
    clip_model="ViT-B/16",
    # fmt: off
    class_name=(
        # nuScenes
        "barrier", "bicycle", "bus", "car", "construction vehicle",
        "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
        "path suitable or safe for driving", "other flat", "sidewalk", "terrain", "man made", "vegetation",
    ),
    valid_index=(
        [i for i in range(16)],
    ),
)

# scheduler settings
epoch = 24
eval_epoch = 24
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.4,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

data = dict(
    num_classes=16,
    ignore_index=-1,
    names=[
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ],
    train=dict(
        type="NuScenesDataset",
        split="train",
        data_root="data/nuscenes",
        transform=[
            dict(
                type="RandomRotate",
                angle=[-0.25, 0.25],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
                keys=["lidar2img", "lidar2cam"],
            ),
            dict(
                type="RandomScale",
                scale=[0.9, 1.1],
                anisotropic=False,
                keys=["lidar2img", "lidar2cam"],
            ),
            dict(
                type="RandomShift",
                shift=[0.5, 0.5, 0.5],
                keys=["lidar2img", "lidar2cam"],
            ),
            dict(
                type="RandomFlip",
                p=0.5,
                keys=["lidar2img", "lidar2cam"],
            ),
            dict(
                type="PointRangeFilter",
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                padding=0.1,
            ),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="ravel",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(
                type="ProjectOnImage",
                filter_overlap=True,
                close_radius=3.0,
            ),
            dict(
                type="RaySample",
                point_nsample=512,
                fetch_color=False,
                fetch_segment=True,
            ),
            dict(type="Add", keys_dict={"condition": "nuScenes"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "condition",
                    "ray_start",
                    "ray_end",
                    "ray_segment",
                ),
                offset_keys_dict=dict(offset="coord", ray_offset="ray_start"),
                stack_keys=("lidar2img", "lidar2cam", "cam_intrinsic"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1,
        use_camera=True,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
