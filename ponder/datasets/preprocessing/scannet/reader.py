# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import os
import sys
import zipfile
from glob import glob

import cv2
import imageio.v2 as imageio
import numpy as np

from ponder.datasets.preprocessing.scannet.SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument("--scans_path", required=True, help="path to scans folder")
parser.add_argument("--output_path", required=True, help="path to output folder")
parser.add_argument(
    "--export_depth_images", dest="export_depth_images", action="store_true"
)
parser.add_argument(
    "--export_color_images", dest="export_color_images", action="store_true"
)
parser.add_argument("--export_poses", dest="export_poses", action="store_true")
parser.add_argument(
    "--export_intrinsics", dest="export_intrinsics", action="store_true"
)
parser.add_argument("--export_label", dest="export_label", action="store_true")
parser.set_defaults(
    export_depth_images=False,
    export_color_images=False,
    export_poses=False,
    export_intrinsics=False,
    export_label=False,
)

opt = parser.parse_args()
print(opt)


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    # assert os.path.isfile(filename)
    mapping = dict()
    # print(filename)
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def main():
    scans = glob(opt.scans_path + "/*")
    scans.sort()

    label_mapping = None
    if opt.export_label:
        root = os.path.dirname(opt.scans_path)
        label_map = read_label_mapping(
            filename=os.path.join(root, "scannetv2-labels.combined.tsv"),
            label_from="id",
            label_to="nyu40id",
        )

    for scan in scans:
        scenename = scan.split("/")[-1]
        filename = os.path.join(scan, scenename + ".sens")
        if not os.path.exists(opt.output_path):
            os.makedirs(opt.output_path)
            # os.makedirs(os.path.join(opt.output_path, 'depth'))
            # os.makedirs(os.path.join(opt.output_path, 'color'))
            # os.makedirs(os.path.join(opt.output_path, 'pose'))
            # os.makedirs(os.path.join(opt.output_path, 'intrinsic'))
            os.makedirs(os.path.join(opt.output_path, scenename))
            # load the data
        print("loading %s..." % filename)
        sd = SensorData(filename)
        print("loaded!\n")
        if opt.export_depth_images:
            # sd.export_depth_images(os.path.join(opt.output_path, 'depth', scenename))
            sd.export_depth_images(os.path.join(opt.output_path, scenename, "depth"))
        if opt.export_color_images:
            # sd.export_color_images(os.path.join(opt.output_path, 'color', scenename))
            sd.export_color_images(os.path.join(opt.output_path, scenename, "color"))
        if opt.export_poses:
            # sd.export_poses(os.path.join(opt.output_path, 'pose', scenename))
            sd.export_poses(os.path.join(opt.output_path, scenename, "pose"))
        if opt.export_intrinsics:
            # sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic', scenename))
            sd.export_intrinsics(os.path.join(opt.output_path, scenename, "intrinsic"))

        os.system(f"cp {scan}/scene*.txt {opt.output_path}/{scenename}/")

        if opt.export_label:

            def map_label_image(image, label_mapping):
                mapped = np.copy(image)
                for k, v in label_mapping.items():
                    mapped[image == k] = v
                return mapped.astype(np.uint8)

            label_zip_path = os.path.join(
                opt.scans_path, scenename, f"{scenename}_2d-label-filt.zip"
            )
            print("process labels")
            with open(label_zip_path, "rb") as f:
                zip_file = zipfile.ZipFile(f)
                for frame in range(0, len(sd.frames)):
                    label_file = f"label-filt/{frame}.png"
                    with zip_file.open(label_file) as lf:
                        image = np.array(imageio.imread(lf))

                    mapped_image = map_label_image(image, label_map)
                    output_path = os.path.join(opt.output_path, scenename, "label")
                    os.makedirs(output_path, exist_ok=True)
                    print("output:", output_path)
                    cv2.imwrite(os.path.join(output_path, f"{frame}.png"), mapped_image)


if __name__ == "__main__":
    main()
