"""
Preprocessing Script for S3DIS
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

try:
    import open3d
except ImportError:
    import warnings

    warnings.warn("Please install open3d for parsing normal")

try:
    import trimesh
except ImportError:
    import warnings

    warnings.warn("Please install trimesh for parsing normal")

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from scipy.spatial import KDTree

area_mesh_dict = {}


def unproject_filtering_depths(
    depths,
    camera_matrix,
    extrinsic,
    depth_scale=4000.0,
    room_coords=None,
    room_semantic=None,
):
    depths[np.isnan(depths)] = 0
    depths[depths >= 65500] = 0
    depths /= depth_scale

    semantic_map = np.zeros_like(depths) - 1

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.0).reshape(-1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points.astype(np.float32).reshape(-1, 3)

    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    points = extrinsic @ points[:, :, None]
    points = points / points[:, -1:, :]
    points = points[:, :3]

    kdtree = KDTree(room_coords)
    dists, indices = kdtree.query(points.reshape(-1, 3), workers=-1)
    mask = mask & (dists < 0.1)
    semantic_map = room_semantic[indices].reshape(*semantic_map.shape)

    mask = mask.reshape(-1)
    depth_mask = mask.reshape(depths.shape)
    semantic_map[~depth_mask] = -1

    return depth_mask, semantic_map


def parse_room(
    room,
    angle,
    dataset_root,
    raw_root,
    output_root,
    align_angle=True,
    parse_normal=False,
    parse_rgbd=False,
    plugin_rgbd=False,
):
    print("Parsing: {}".format(room))

    if not plugin_rgbd:
        classes = [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",
        ]
        class2label = {cls: i for i, cls in enumerate(classes)}
        source_dir = os.path.join(dataset_root, room)
        save_path = os.path.join(output_root, room) + ".pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        object_path_list = sorted(
            glob.glob(os.path.join(source_dir, "Annotations/*.txt"))
        )

        room_coords = []
        room_colors = []
        room_normals = []
        room_semantic_gt = []
        room_instance_gt = []

        for object_id, object_path in enumerate(object_path_list):
            object_name = os.path.basename(object_path).split("_")[0]
            obj = np.loadtxt(object_path)
            coords = obj[:, :3]
            colors = obj[:, 3:6]
            # note: in some room there is 'stairs' class
            class_name = object_name if object_name in classes else "clutter"
            semantic_gt = np.repeat(class2label[class_name], coords.shape[0])
            semantic_gt = semantic_gt.reshape([-1, 1])
            instance_gt = np.repeat(object_id, coords.shape[0])
            instance_gt = instance_gt.reshape([-1, 1])

            room_coords.append(coords)
            room_colors.append(colors)
            room_semantic_gt.append(semantic_gt)
            room_instance_gt.append(instance_gt)

        room_coords = np.ascontiguousarray(np.vstack(room_coords))

        if parse_normal:
            x_min, z_max, y_min = np.min(room_coords, axis=0)
            x_max, z_min, y_max = np.max(room_coords, axis=0)
            z_max = -z_max
            z_min = -z_min
            max_bound = np.array([x_max, y_max, z_max]) + 0.1
            min_bound = np.array([x_min, y_min, z_min]) - 0.1
            bbox = open3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound, max_bound=max_bound
            )
            # crop room
            room_mesh = (
                area_mesh_dict[os.path.dirname(room)]
                .crop(bbox)
                .transform(
                    np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                )
            )
            vertices = np.array(room_mesh.vertices)
            faces = np.array(room_mesh.triangles)
            vertex_normals = np.array(room_mesh.vertex_normals)
            room_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=vertex_normals
            )
            (closest_points, distances, face_id) = room_mesh.nearest.on_surface(
                room_coords
            )
            room_normals = room_mesh.face_normals[face_id]

        if align_angle:
            angle = (2 - angle / 180) * np.pi
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            room_center = (
                np.max(room_coords, axis=0) + np.min(room_coords, axis=0)
            ) / 2
            room_coords = (room_coords - room_center) @ np.transpose(
                rot_t
            ) + room_center
            if parse_normal:
                room_normals = room_normals @ np.transpose(rot_t)

        room_colors = np.ascontiguousarray(np.vstack(room_colors))
        room_semantic_gt = np.ascontiguousarray(np.vstack(room_semantic_gt))
        room_instance_gt = np.ascontiguousarray(np.vstack(room_instance_gt))
        save_dict = dict(
            coord=room_coords,
            color=room_colors,
            semantic_gt=room_semantic_gt,
            instance_gt=room_instance_gt,
            room_center=room_center,
        )
        if parse_normal:
            save_dict["normal"] = room_normals
        torch.save(save_dict, save_path)
    else:
        save_path = os.path.join(output_root, room) + ".pth"
        save_dict = torch.load(save_path)
        room_coords = save_dict["coord"]
        room_colors = save_dict["color"]
        room_semantic_gt = save_dict["semantic_gt"]
        if align_angle:
            angle = (2 - angle / 180) * np.pi
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            room_center = save_dict["room_center"]

    if parse_rgbd or plugin_rgbd:
        area_indices = int(room.split("/")[0].split("_")[1])
        area_indices = [area_indices] if area_indices != 5 else ["5a", "5b"]
        for area_index in area_indices:
            cam2room = json.load(
                open(
                    os.path.join(
                        raw_root, f"area_{area_index}", "3d/camera_to_room.json"
                    )
                )
            )
            room2cam = defaultdict(list)
            for k, v in cam2room.items():
                room2cam[v].append(k)
            uuids = room2cam[
                f"{room.split('/')[1]}_{area_index if isinstance(area_index, int) else int(area_index[0])}"
            ]
            for uuid in uuids:
                pose_paths = glob.glob(
                    os.path.join(
                        raw_root, f"area_{area_index}", "raw", f"{uuid}_pose_*.txt"
                    )
                )
                for pose_path in pose_paths:
                    try:
                        uuid, _, pitch_level, yaw_position = (
                            os.path.basename(pose_path).split(".")[0].split("_")
                        )
                        intrinsic_path = os.path.join(
                            raw_root,
                            f"area_{area_index}",
                            "raw",
                            f"{uuid}_intrinsics_{pitch_level}.txt",
                        )
                        h, w, fx, fy, cx, cy, k1, k2, p1, p2, k3 = np.loadtxt(
                            intrinsic_path
                        )
                        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        pose = np.loadtxt(pose_path)

                        if area_index == "5b":
                            pose = (
                                np.array(
                                    [
                                        [0, 1, 0, -4.09703582],
                                        [-1, 0, 0, 6.22617759],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1],
                                    ]
                                )
                                @ pose
                            )
                        if align_angle:
                            S_1 = np.eye(4)
                            S_2 = np.eye(4)
                            S_3 = np.eye(4)
                            S_1[:3, 3] = -room_center
                            S_2[:3, :3] = rot_t
                            S_3[:3, 3] = room_center
                            S = S_3 @ S_2 @ S_1
                            pose = S @ pose

                        rgb = cv2.cvtColor(
                            cv2.imread(
                                os.path.join(
                                    raw_root,
                                    f"area_{area_index}",
                                    "raw",
                                    f"{uuid}_i{pitch_level}_{yaw_position}.jpg",
                                )
                            ),
                            cv2.COLOR_BGR2RGB,
                        )
                        depth = cv2.imread(
                            os.path.join(
                                raw_root,
                                f"area_{area_index}",
                                "raw",
                                f"{uuid}_d{pitch_level}_{yaw_position}.png",
                            ),
                            cv2.IMREAD_UNCHANGED,
                        )
                        undistorted_rgb = cv2.undistort(
                            rgb, intrinsic, np.array([k1, k2, p1, p2, k3])
                        )
                        undistorted_depth = cv2.undistort(
                            depth, intrinsic, np.array([k1, k2, p1, p2, k3])
                        )
                        depth_mask, semantic_map = unproject_filtering_depths(
                            undistorted_depth.astype(float),
                            intrinsic,
                            pose,
                            depth_scale=4000.0,
                            room_coords=room_coords,
                            room_semantic=room_semantic_gt,
                        )

                        rgbd_dict = dict(
                            intrinsic=intrinsic,
                            extrinsic=pose,
                            rgb=undistorted_rgb,
                            depth=undistorted_depth,
                            depth_mask=depth_mask,
                            semantic_map=semantic_map,
                        )

                        rgbd_save_path = os.path.join(
                            output_root,
                            f"{room}_rgbd",
                            f"{uuid}_{pitch_level}_{yaw_position}.pth",
                        )
                        os.makedirs(os.path.dirname(rgbd_save_path), exist_ok=True)
                        torch.save(rgbd_dict, rgbd_save_path)
                    except Exception as e:
                        print(f"Skip {pose_path}. Error: {e}")
                        continue


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to Stanford3dDataset_v1.2 dataset"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where area folders will be located",
    )
    parser.add_argument(
        "--raw_root",
        default=None,
        help="Path to Stanford2d3dDataset_noXYZ dataset (optional)",
    )
    parser.add_argument(
        "--align_angle", action="store_true", help="Whether align room angles"
    )
    parser.add_argument(
        "--parse_normal", action="store_true", help="Whether process normal"
    )
    parser.add_argument(
        "--parse_rgbd",
        action="store_true",
        help="Whether to parse RGB-D images.",
        default=True,
    )
    parser.add_argument(
        "--plugin_rgbd",
        action="store_true",
        help="Whether to parse RGB-D images only as a plugin mode.",
        default=True,
    )
    args = parser.parse_args()

    if args.parse_normal:
        assert args.raw_root is not None

    room_list = []
    angle_list = []

    # Load room information
    print("Loading room information ...")
    for i in range(1, 7):
        area_info = np.loadtxt(
            os.path.join(
                args.dataset_root,
                "Area_{}".format(i),
                "Area_{}_alignmentAngle.txt".format(i),
            ),
            dtype=str,
        )
        room_list += [
            os.path.join("Area_{}".format(i), room_info[0]) for room_info in area_info
        ]
        angle_list += [int(room_info[1]) for room_info in area_info]

    if args.parse_normal:
        # load raw mesh file to extract normal
        print("Loading raw mesh file ...")
        for i in range(1, 7):
            if i != 5:
                mesh_dir = os.path.join(
                    args.raw_root, "area_{}".format(i), "3d", "rgb.obj"
                )
                mesh = open3d.io.read_triangle_mesh(mesh_dir)
                if hasattr(mesh.triangle_uvs, "clear"):
                    mesh.triangle_uvs.clear()
                else:  # for open3d <= 0.10.0
                    for _ in range(len(mesh.triangle_uvs)):
                        tmp = mesh.triangle_uvs.pop()
            else:
                mesh_a_dir = os.path.join(
                    args.raw_root, "area_{}a".format(i), "3d", "rgb.obj"
                )
                mesh_b_dir = os.path.join(
                    args.raw_root, "area_{}b".format(i), "3d", "rgb.obj"
                )
                mesh_a = open3d.io.read_triangle_mesh(mesh_a_dir)

                if hasattr(mesh_a.triangle_uvs, "clear"):
                    mesh_a.triangle_uvs.clear()
                else:  # for open3d <= 0.10.0
                    for _ in range(len(mesh_a.triangle_uvs)):
                        tmp = mesh_a.triangle_uvs.pop()

                mesh_b = open3d.io.read_triangle_mesh(mesh_b_dir)
                if hasattr(mesh_b.triangle_uvs, "clear"):
                    mesh_b.triangle_uvs.clear()
                else:  # for open3d <= 0.10.0
                    for _ in range(len(mesh_b.triangle_uvs)):
                        tmp = mesh_b.triangle_uvs.pop()

                mesh_b = mesh_b.transform(
                    np.array(
                        [
                            [0, 0, -1, -4.09703582],
                            [0, 1, 0, 0],
                            [1, 0, 0, -6.22617759],
                            [0, 0, 0, 1],
                        ]
                    )
                )
                try:
                    mesh = mesh_a + mesh_b
                except:  # for open3d <= 0.10.0
                    mesh = open3d.geometry.TriangleMesh(
                        mesh_a.vertices, mesh_a.triangles
                    ) + open3d.geometry.TriangleMesh(mesh_b.vertices, mesh_b.triangles)
                    mesh.compute_vertex_normals()

            area_mesh_dict["Area_{}".format(i)] = mesh
            print("Area_{} mesh is loaded".format(i))

    # Preprocess data.
    print("Processing scenes...")
    # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    pool = ProcessPoolExecutor(max_workers=8)  # peak 110G memory when parsing normal.
    _ = list(
        pool.map(
            parse_room,
            room_list,
            angle_list,
            repeat(args.dataset_root),
            repeat(args.raw_root),
            repeat(args.output_root),
            repeat(args.align_angle),
            repeat(args.parse_normal),
            repeat(args.parse_rgbd),
            repeat(args.plugin_rgbd),
        )
    )


if __name__ == "__main__":
    main_process()
