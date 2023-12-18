"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

from ponder.utils.cache import shared_dict
from ponder.utils.logger import get_root_logger

from .builder import DATASETS
from .transform import TRANSFORMS, Compose


@DATASETS.register_module()
class S3DISDataset(Dataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(S3DISDataset, self).__init__()
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

        self.data_list = self.get_data_list()
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
        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data_path
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
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
class S3DISRGBDDataset(S3DISDataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        num_cameras=5,
        render_semantic=True,
        six_fold=False,
        loop=1,
    ):
        super(S3DISRGBDDataset, self).__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            cache=cache,
            loop=loop,
        )
        self.num_cameras = num_cameras
        self.render_semantic = render_semantic
        self.six_fold = six_fold

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError

        print("Filtering S3DIS RGBD dataset...")
        filtered_data_list = []
        for data_path in data_list:
            rgbd_paths = glob.glob(
                os.path.join(data_path.split(".pth")[0] + "_rgbd", "*.pth")
            )
            if len(rgbd_paths) <= 0:
                # print(f"{data_path} has no rgbd data.")
                continue
            filtered_data_list.append(data_path)
        print(
            f"Finish filtering! Totally {len(filtered_data_list)} from {len(data_list)} data."
        )
        return filtered_data_list

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

        rgbd_paths = glob.glob(
            os.path.join(data_path.split(".pth")[0] + "_rgbd", "*.pth")
        )

        if len(rgbd_paths) <= 0:
            print(f"{data_path} has no rgbd data.")
            return self.get_data(np.random.randint(0, self.__len__()))

        rgbd_paths = np.random.choice(
            rgbd_paths, self.num_cameras, replace=self.num_cameras > len(rgbd_paths)
        )
        rgbd_dicts = [torch.load(p) for p in rgbd_paths]

        for i in range(len(rgbd_dicts)):
            if (rgbd_dicts[i]["depth_mask"]).mean() < 0.25:
                return self.get_data(idx)

        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data_path
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            intrinsic=np.stack([d["intrinsic"] for d in rgbd_dicts], axis=0).astype(
                np.float32
            ),
            extrinsic=np.stack(
                [np.linalg.inv(d["extrinsic"]) for d in rgbd_dicts], axis=0
            ).astype(np.float32),
            rgb=np.stack([d["rgb"].astype(np.float32) for d in rgbd_dicts], axis=0),
            depth=np.stack(
                [
                    d["depth"].astype(np.float32)
                    * d["depth_mask"].astype(np.float32)
                    * (d["depth"] < 65535).astype(np.float32)
                    for d in rgbd_dicts
                ],
                axis=0,
            ),
            depth_scale=1.0 / 4000.0,
        )

        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
        if self.render_semantic:
            for d in rgbd_dicts:
                d["semantic_map"][d["semantic_map"] <= 0] = -1
                d["semantic_map"][d["semantic_map"] > 40] = -1
                d["semantic_map"] = d["semantic_map"].astype(np.int16)
            data_dict.update(
                dict(semantic=np.stack([d["semantic_map"] for d in rgbd_dicts], axis=0))
            )
        if (
            self.six_fold
        ):  # pretrain for 6-fold cross validation, ignore semantic labels to avoid leaking information
            data_dict["semantic"] = np.zeros_like(data_dict["semantic"]) - 1
        return data_dict
