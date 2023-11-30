import argparse
import os
import pickle
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}


def get_available_scenes(nusc):
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor"s coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def obtain_sensor2top(nusc, sensor_token, ego2lidar, global2ego, sensor_type="lidar"):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        sensor_type (str): Sensor to calibrate. Default: "lidar".

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    sensor2ego = transform_matrix(
        cs_rec["translation"], Quaternion(cs_rec["rotation"]), inverse=False
    )
    ego2global = transform_matrix(
        pose_rec["translation"], Quaternion(pose_rec["rotation"]), inverse=False
    )
    sensor2lidar = reduce(np.dot, [ego2lidar, global2ego, ego2global, sensor2ego])
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "ego2global": ego2global,
        "sensor2ego": sensor2ego,
        "sensor2lidar": sensor2lidar,
        "timestamp": 1e-6 * sd_rec["timestamp"],
    }
    return sweep


def fill_trainval_infos(
    data_path, nusc, train_scenes, test=False, max_sweeps=10, with_camera=False
):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(
        total=len(nusc.sample), desc="create_info", dynamic_ncols=True
    )

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        assert sd_rec["timestamp"] == sample["timestamp"]
        lidar_path, boxes, _ = get_sample_data(nusc, lidar_token)

        ego2lidar = transform_matrix(
            cs_rec["translation"], Quaternion(cs_rec["rotation"]), inverse=True
        )
        global2ego = transform_matrix(
            pose_rec["translation"], Quaternion(pose_rec["rotation"]), inverse=True
        )
        info = {
            "lidar_path": Path(lidar_path).relative_to(data_path).__str__(),
            "lidar_token": lidar_token,
            "token": sample["token"],
            "sweeps": [],
            "ego2lidar": ego2lidar,
            "global2ego": global2ego,
            "timestamp": 1e-6 * sd_rec["timestamp"],  # TODO: check
        }
        if with_camera:
            info["cams"] = dict()
            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, ego2lidar, global2ego, cam
                )
                cam_info["data_path"] = (
                    Path(cam_info["data_path"]).relative_to(data_path).__str__()
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc, sd_rec["prev"], ego2lidar, global2ego, "lidar"
                )
                sweep["data_path"] = (
                    Path(sweep["data_path"]).relative_to(data_path).__str__()
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps

        if with_camera:
            # obtain image sweeps for a single key-frame
            info["cam_sweeps"] = dict()
            for cam in camera_types:
                cam_rec = nusc.get("sample_data", sample["data"][cam])
                cam_sweeps = []
                # max sweep for camera is actually 6
                while len(cam_sweeps) < max_sweeps:
                    if not cam_rec["prev"] == "":
                        cam_token = cam_rec["token"]
                        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                        cam_sweep = obtain_sensor2top(
                            nusc, cam_token, ego2lidar, global2ego, cam
                        )
                        cam_sweep["data_path"] = (
                            Path(cam_sweep["data_path"])
                            .relative_to(data_path)
                            .__str__()
                        )
                        cam_sweep.update(cam_intrinsic=cam_intrinsic)
                        cam_sweeps.append(cam_sweep)
                        cam_rec = nusc.get("sample_data", cam_rec["prev"])
                    else:
                        break
                info["cam_sweeps"][cam] = cam_sweeps

        if not test:
            # processing gt bbox
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno["num_lidar_pts"] for anno in annotations])
            num_radar_pts = np.array([anno["num_radar_pts"] for anno in annotations])
            mask = num_lidar_pts + num_radar_pts > 0

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)[
                :, [1, 0, 2]
            ]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in boxes]).reshape(
                -1, 1
            )
            names = np.array([b.name for b in boxes])
            tokens = np.array([b.token for b in boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info["gt_boxes"] = gt_boxes[mask, :]
            info["gt_boxes_velocity"] = velocity[mask, :]
            info["gt_names"] = np.array(
                [map_name_from_general_to_detection[name] for name in names]
            )[mask]
            info["gt_boxes_token"] = tokens[mask]
            info["num_lidar_pts"] = num_lidar_pts[mask]
            info["num_radar_pts"] = num_radar_pts[mask]

            # processing gt segment
            info["gt_segment_path"] = nusc.get("lidarseg", lidar_token)["filename"]

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to the nuScenes dataset."
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where processed information located.",
    )
    parser.add_argument(
        "--max_sweeps", default=10, type=int, help="Max number of sweeps. Default: 10."
    )
    parser.add_argument(
        "--with_camera",
        action="store_true",
        default=False,
        help="Whether use camera or not.",
    )
    config = parser.parse_args()

    print(f"Loading nuScenes tables for version v1.0-trainval...")
    nusc_trainval = NuScenes(
        version="v1.0-trainval", dataroot=config.dataset_root, verbose=False
    )
    available_scenes_trainval = get_available_scenes(nusc_trainval)
    available_scene_names_trainval = [s["name"] for s in available_scenes_trainval]
    print("total scene num:", len(nusc_trainval.scene))
    print("exist scene num:", len(available_scenes_trainval))
    assert len(available_scenes_trainval) == len(nusc_trainval.scene) == 850

    print(f"Loading nuScenes tables for version v1.0-test...")
    nusc_test = NuScenes(
        version="v1.0-test", dataroot=config.dataset_root, verbose=False
    )
    available_scenes_test = get_available_scenes(nusc_test)
    available_scene_names_test = [s["name"] for s in available_scenes_test]
    print("total scene num:", len(nusc_test.scene))
    print("exist scene num:", len(available_scenes_test))
    assert len(available_scenes_test) == len(nusc_test.scene) == 150

    train_scenes = splits.train
    train_scenes = set(
        [
            available_scenes_trainval[available_scene_names_trainval.index(s)]["token"]
            for s in train_scenes
        ]
    )
    test_scenes = splits.test
    test_scenes = set(
        [
            available_scenes_test[available_scene_names_test.index(s)]["token"]
            for s in test_scenes
        ]
    )
    print(f"Filling trainval information...")
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        config.dataset_root,
        nusc_trainval,
        train_scenes,
        test=False,
        max_sweeps=config.max_sweeps,
        with_camera=config.with_camera,
    )
    print(f"Filling test information...")
    test_nusc_infos, _ = fill_trainval_infos(
        config.dataset_root,
        nusc_test,
        test_scenes,
        test=True,
        max_sweeps=config.max_sweeps,
        with_camera=config.with_camera,
    )

    print(f"Saving nuScenes information...")
    os.makedirs(os.path.join(config.output_root, "info"), exist_ok=True)
    print(
        f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}, test sample: {len(test_nusc_infos)}"
    )
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_train.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(train_nusc_infos, f)
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_val.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(val_nusc_infos, f)
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_test.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(test_nusc_infos, f)
