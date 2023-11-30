import os
import pickle
from collections.abc import Sequence

import cv2
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/nuscenes",
        sweeps=10,
        use_camera=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.use_camera = use_camera
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_camera_data(self, data):
        img_list = []
        ori_shape_list = []
        lidar2img_list = []
        lidar2cam_list = []
        cam_intrinsic_list = []

        for cam_type, cam_info in data["cams"].items():
            img = cv2.imread(
                os.path.join(self.data_root, "raw", cam_info["data_path"]),
                cv2.IMREAD_COLOR,
            ).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
            ori_shape_list.append(img.shape)
            # lidar to camera transform
            lidar2cam = np.linalg.inv(cam_info["sensor2lidar"])
            lidar2cam_list.append(lidar2cam)
            # camera intrinsics
            cam_intrinsic = np.eye(4)
            cam_intrinsic[:3, :3] = cam_info["cam_intrinsic"]
            cam_intrinsic_list.append(cam_intrinsic)
            # lidar to image transform
            lidar2img = cam_intrinsic @ lidar2cam
            lidar2img_list.append(lidar2img)

        img_dict = {
            "img": np.stack(img_list, axis=0),
            "ori_shape": np.stack(ori_shape_list, axis=0),
            "lidar2img": np.stack(lidar2img_list, axis=0),
            "lidar2cam": np.stack(lidar2cam_list, axis=0),
            "cam_intrinsic": np.stack(cam_intrinsic_list, axis=0),
        }
        return img_dict

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(coord=coord, strength=strength, segment=segment)

        if self.use_camera:
            img_dict = self.get_camera_data(data)
            data_dict.update(img_dict)

        data_dict["lidar_token"] = data["lidar_token"]
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
