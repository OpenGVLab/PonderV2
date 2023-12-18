"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ponder.utils.cache import shared_dict
from ponder.utils.logger import get_root_logger

from .builder import DATASETS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from .transform import TRANSFORMS, Compose


@DATASETS.register_module()
class ScanNetDataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(ScanNetDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        if lr_file:
            self.data_list = [
                os.path.join(data_root, "train", name + ".pth")
                for name in np.loadtxt(lr_file, dtype=str)
            ]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "ponder" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    class2id = np.array(VALID_CLASS_IDS_200)

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt200" in data.keys():
            segment = data["semantic_gt200"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            segment[sampled_index] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict


@DATASETS.register_module()
class ScanNetRGBDDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        rgbd_root="data/scannet/rgbd",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        frame_interval=10,
        nearby_num=2,
        nearby_interval=20,
        num_cameras=5,
        render_semantic=True,
        align_axis=False,
        loop=1,
    ):
        super(ScanNetRGBDDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.rgbd_root = rgbd_root
        self.frame_interval = frame_interval
        self.nearby_num = nearby_num
        self.nearby_interval = nearby_interval
        self.num_cameras = num_cameras
        self.render_semantic = render_semantic
        self.align_axis = align_axis

        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.logger = get_root_logger()

        if lr_file:
            full_data_list = self.get_data_list()
            self.data_list = []
            lr_list = np.loadtxt(lr_file, dtype=str)
            for data_dict in full_data_list:
                if data_dict["scene"] in lr_list:
                    self.data_list.append(data_dict)
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index

        self.logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        self.axis_align_matrix_list = {}
        self.intrinsic_list = {}
        self.frame_lists = {}

        # Get all models
        data_list = []
        split_json = os.path.join(os.path.join(self.data_root, self.split + ".json"))

        if os.path.exists(split_json):
            with open(split_json, "r") as f:
                data_list = json.load(f)
        else:
            scene_list = [
                filename.split(".")[0]
                for filename in os.listdir(os.path.join(self.data_root, self.split))
            ]

            skip_list = []
            skip_counter = 0
            skip_file = os.path.join(os.path.join(self.data_root, "skip.lst"))
            if os.path.exists(skip_file):
                with open(skip_file, "r") as f:
                    for i in f.read().split("\n"):
                        scene_name, frame_idx = i.split()
                        skip_list.append((scene_name, int(frame_idx)))

            # walk through the subfolder
            from tqdm import tqdm

            for scene_name in tqdm(scene_list):
                # filenames = os.listdir(os.path.join(subpath, m, 'pointcloud'))
                frame_list = self.get_frame_list(scene_name)

                # for test and val, we only use 1/10 of the data, since those data will not affect
                # the training and we use them just for visualization and debugging
                if self.split == "val":
                    frame_list = frame_list[::10]
                if self.split == "test":
                    frame_list = frame_list[::10]

                for frame_idx in frame_list[
                    self.nearby_num
                    * self.nearby_interval : -(self.nearby_num + 1)
                    * self.nearby_interval : self.frame_interval
                ]:
                    frame_idx = int(frame_idx.split(".")[0])
                    if (scene_name, frame_idx) in skip_list:
                        skip_counter += 1
                        continue
                    data_list.append({"scene": scene_name, "frame": frame_idx})

            self.logger.info(
                f"ScanNet: <{skip_counter} Frames will be skipped in {self.split} data.>"
            )

            with open(split_json, "w") as f:
                json.dump(data_list, f)

        data_dict = defaultdict(list)
        for data in data_list:
            data_dict[data["scene"]].append(data["frame"])

        data_list = []
        for scene_name, frame_list in data_dict.items():
            data_list.append({"scene": scene_name, "frame": frame_list})

        return data_list

    def get_data(self, idx):
        scene_name = self.data_list[idx % len(self.data_list)]["scene"]
        frame_list = self.data_list[idx % len(self.data_list)]["frame"]
        scene_path = os.path.join(self.data_root, self.split, f"{scene_name}.pth")
        if not self.cache:
            data = torch.load(scene_path)
        else:
            data_name = scene_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "ponder" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)

        if self.num_cameras > len(frame_list):
            print(
                f"Warning: {scene_name} has only {len(frame_list)} frames, "
                f"but {self.num_cameras} cameras are required."
            )
        frame_idxs = np.random.choice(
            frame_list, self.num_cameras, replace=self.num_cameras > len(frame_list)
        )
        intrinsic, extrinsic, rgb, depth = (
            [],
            [],
            [],
            [],
        )

        if self.render_semantic:
            semantic = []
        for frame_idx in frame_idxs:
            if not self.render_semantic:
                intri, rot, transl, rgb_im, depth_im = self.get_2d_meta(
                    scene_name, frame_idx
                )
            else:
                intri, rot, transl, rgb_im, depth_im, semantic_im = self.get_2d_meta(
                    scene_name, frame_idx
                )
                assert semantic_im.max() <= 20, semantic_im
                semantic.append(semantic_im)
            intrinsic.append(intri)
            extri = np.eye(4)
            extri[:3, :3] = rot
            extri[:3, 3] = transl
            extrinsic.append(extri)
            rgb.append(rgb_im)
            depth.append(depth_im)

        intrinsic = np.stack(intrinsic, axis=0)
        extrinsic = np.stack(extrinsic, axis=0)
        rgb = np.stack(rgb, axis=0)
        depth = np.stack(depth, axis=0)

        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            rgb=rgb,
            depth=depth,
            depth_scale=1.0 / 1000.0,
            id=f"{scene_name}/{frame_idxs[0]}",
        )
        if self.render_semantic:
            semantic = np.stack(semantic, axis=0)
            data_dict.update(dict(semantic=semantic))

        if self.la:
            sampled_index = self.la[self.get_data_name(scene_path)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
            data_dict["semantic"] = np.zeros_like(data_dict["semantic"]) - 1

        return data_dict

    def get_data_name(self, scene_path):
        return os.path.basename(scene_path).split(".")[0]

    def get_frame_list(self, scene_name):
        if scene_name in self.frame_lists:
            return self.frame_lists[scene_name]

        if not os.path.exists(os.path.join(self.rgbd_root, scene_name, "color")):
            return []

        frame_list = os.listdir(os.path.join(self.rgbd_root, scene_name, "color"))
        frame_list = list(frame_list)
        frame_list = [frame for frame in frame_list if frame.endswith(".jpg")]
        frame_list.sort(key=lambda x: int(x.split(".")[0]))
        self.frame_lists[scene_name] = frame_list
        return self.frame_lists[scene_name]

    def get_axis_align_matrix(self, scene_name):
        if scene_name in self.axis_align_matrix_list:
            return self.axis_align_matrix_list[scene_name]
        txt_file = os.path.join(self.rgbd_root, scene_name, "%s.txt" % scene_name)
        # align axis
        with open(txt_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "axisAlignment" in line:
                self.axis_align_matrix_list[scene_name] = [
                    float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
                ]
                break
        self.axis_align_matrix_list[scene_name] = np.array(
            self.axis_align_matrix_list[scene_name]
        ).reshape((4, 4))
        return self.axis_align_matrix_list[scene_name]

    def get_intrinsic(self, scene_name):
        if scene_name in self.intrinsic_list:
            return self.intrinsic_list[scene_name]
        self.intrinsic_list[scene_name] = np.loadtxt(
            os.path.join(self.rgbd_root, scene_name, "intrinsic", "intrinsic_depth.txt")
        )
        return self.intrinsic_list[scene_name]

    def get_2d_meta(self, scene_name, frame_idx):
        # framelist
        frame_list = self.get_frame_list(scene_name)
        intrinsic = self.get_intrinsic(scene_name)
        if self.align_axis:
            axis_align_matrix = self.get_axis_align_matrix(scene_name)

        if not self.render_semantic:
            rgb_im, depth_im, pose = self.read_data(scene_name, frame_list[frame_idx])
        else:
            rgb_im, depth_im, pose, semantic_im = self.read_data(
                scene_name, frame_list[frame_idx]
            )
            semantic_im_40 = cv2.resize(
                semantic_im,
                (depth_im.shape[1], depth_im.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            semantic_im_40 = semantic_im_40.astype(np.int16)
            semantic_im = np.zeros_like(semantic_im_40) - 1
            for i, id in enumerate(VALID_CLASS_IDS_20):
                semantic_im[semantic_im_40 == id] = i

        rgb_im = cv2.resize(rgb_im, (depth_im.shape[1], depth_im.shape[0]))
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)  # H, W, 3
        depth_im = depth_im.astype(np.float32)  # H, W

        if self.align_axis:
            pose = np.matmul(axis_align_matrix, pose)
        pose = np.linalg.inv(pose)

        intrinsic = np.array(intrinsic)
        rotation = np.array(pose)[:3, :3]
        translation = np.array(pose)[:3, 3]

        if not self.render_semantic:
            return intrinsic, rotation, translation, rgb_im, depth_im
        else:
            return intrinsic, rotation, translation, rgb_im, depth_im, semantic_im

    def read_data(self, scene_name, frame_name):
        color_path = os.path.join(self.rgbd_root, scene_name, "color", frame_name)
        depth_path = os.path.join(
            self.rgbd_root, scene_name, "depth", frame_name.replace(".jpg", ".png")
        )

        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb_im = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)

        pose = np.loadtxt(
            os.path.join(
                self.rgbd_root,
                scene_name,
                "pose",
                frame_name.replace(".jpg", ".txt"),
            )
        )

        if not self.render_semantic:
            return rgb_im, depth_im, pose
        else:
            seg_path = os.path.join(
                self.rgbd_root,
                scene_name,
                "label",
                frame_name.replace(".jpg", ".png"),
            )
            semantic_im = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            return rgb_im, depth_im, pose, semantic_im

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
