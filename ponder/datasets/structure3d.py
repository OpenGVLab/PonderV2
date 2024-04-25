"""
Structured3D Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
from collections.abc import Sequence

import numpy as np
import torch

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*/*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*/*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        scene_name = os.path.basename(dir_path)
        room_name = os.path.splitext(file_name)[0]
        data_name = f"{scene_name}_{room_name}"
        return data_name


@DATASETS.register_module()
class Structured3DRGBDDataset(Structured3DDataset):
    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        num_cameras=5,
        render_semantic=True,
        loop=1,
    ):
        super(Structured3DRGBDDataset, self).__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )
        self.num_cameras = num_cameras
        self.render_semantic = render_semantic

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*/*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*/*.pth"))
        else:
            raise NotImplementedError

        print("Filtering Structured3D RGBD dataset...")
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
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1

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
                os.rename(rgbd_paths[i], rgbd_paths[i] + ".bad")
                return self.get_data(idx)

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
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
            depth_scale=1.0 / 1000.0,
        )
        if self.render_semantic:
            for d in rgbd_dicts:
                d["semantic_map"][d["semantic_map"] <= 0] = -1
                d["semantic_map"][d["semantic_map"] > 40] = -1
                d["semantic_map"] = d["semantic_map"].astype(np.int16)
            data_dict.update(
                dict(semantic=np.stack([d["semantic_map"] for d in rgbd_dicts], axis=0))
            )
        return data_dict
