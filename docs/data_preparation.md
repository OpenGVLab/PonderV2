## Data Preparation

### ScanNet v2

The preprocessing support semantic and instance segmentation for both ScanNet20, ScanNet200 and ScanNet Data Efficient.

- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Run preprocessing code for raw ScanNet as follows:
```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset (output dir).
python ponder/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
# extract RGB-D iamges and 2D semantic labels:
python ponder/datasets/preprocessing/scannet/reader.py --scans_path ${RAW_SCANNET_DIR}/scans --output_path ${PROCESSED_SCANNET_DIR}/rgbd --export_depth_images --export_color_images --export_poses --export_intrinsics --export_label
```

- (Optional) Download ScanNet Data Efficient files:
```bash
# download-scannet.py is the official download script
# or follow instruction here: https://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation#download
python download-scannet.py --data_efficient -o ${RAW_SCANNET_DIR}
# unzip downloads
cd ${RAW_SCANNET_DIR}/tasks
unzip limited-annotation-points.zip
unzip limited-bboxes.zip
unzip limited-reconstruction-scenes.zip
# copy files to processed dataset folder
cp -r ${RAW_SCANNET_DIR}/tasks ${PROCESSED_SCANNET_DIR}
```

- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

## S3DIS
- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
- Run preprocessing code for S3DIS as follows:
```bash
# S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2 dataset.
# RAW_S3DIS_DIR: the directory of Stanford2d3dDataset_noXYZ dataset. (optional, for parsing normal)
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset (output dir).

# S3DIS with normal vector, RGB-D images and 2D semantic labels
python ponder/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --parse_normal --parse_rgbd
# if you want S3DIS with aligned angle:
python ponder/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --align_angle --parse_normal --parse_rgbd
```
- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
mkdir data
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```

## Structured3D
- Download Structured3D panorama related and perspective (full) related zip files by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSc0qtvh4vHSoZaW6UvlXYy79MbcGdZfICjh4_t4bYofQIVIdw/viewform?pli=1) (no need to unzip them).
- Organize all downloaded zip file in one folder (`${STRUCT3D_DIR}`).
- Run preprocessing code for Structured3D as follows:
```bash
# STRUCT3D_DIR: the directory of downloaded Structured3D dataset.
# PROCESSED_STRUCT3D_DIR: the directory of processed Structured3D dataset (output dir).
# NUM_WORKERS: Number for workers for preprocessing, default same as cpu count (might OOM).
export PYTHONPATH=./
python ponder/datasets/preprocessing/structured3d/preprocess_structured3d.py --dataset_root ${STRUCT3D_DIR} --output_root ${PROCESSED_STRUCT3D_DIR} --num_workers ${NUM_WORKERS} --grid_size 0.01 --fuse_prsp --fuse_pano --parse_rgbd
```

Following the instruction of [Swin3D](https://arxiv.org/abs/2304.06906), we keep 25 categories with frequencies of more than 0.001, out of the original 40 categories.

- Link processed dataset to codebase.
```bash
# PROCESSED_STRUCT3D_DIR: the directory of processed Structured3D dataset (output dir).
mkdir data
ln -s ${PROCESSED_STRUCT3D_DIR} ${CODEBASE_DIR}/data/structured3d
```

## nuScenes
- Download the official [NuScene](https://www.nuscenes.org/nuscenes#download) dataset (with Lidar Segmentation) and organize the downloaded files as follows:
```bash
NUSCENES_DIR
│── samples
│── sweeps
│── lidarseg
...
│── v1.0-trainval 
│── v1.0-test
```

- Run information preprocessing code (modified from OpenPCDet) for nuScenes as follows:
```bash
# NUSCENES_DIR: the directory of downloaded nuScenes dataset.
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
# MAX_SWEEPS: Max number of sweeps. Default: 10.
pip install nuscenes-devkit pyquaternion
python ponder/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py --dataset_root ${NUSCENES_DIR} --output_root ${PROCESSED_NUSCENES_DIR} --max_sweeps ${MAX_SWEEPS} --with_camera
```

- Link raw dataset to processed NuScene dataset folder:
```bash
# NUSCENES_DIR: the directory of downloaded nuScenes dataset.
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
ln -s ${NUSCENES_DIR} {PROCESSED_NUSCENES_DIR}/raw
```

then the processed nuscenes folder is organized as follows:
```bash
nuscene
|── raw
    │── samples
    │── sweeps
    │── lidarseg
    ...
    │── v1.0-trainval
    │── v1.0-test
|── info
```

- Link processed dataset to codebase.
```bash
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
mkdir data
ln -s ${PROCESSED_NUSCENES_DIR} ${CODEBASE_DIR}/data/nuscenes
```