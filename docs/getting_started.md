## Getting Started

### (Pre)Training

**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-spunet-v1m1-0-base.py --options save_path=exp/scannet/semseg-spunet-v1m1-0-base
```
**Resume training from checkpoint.** If the training process is interrupted by accident, the following script can resume training from a given checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

### Testing
During training, model evaluation is performed on point clouds after grid sampling (voxelization), providing an initial assessment of model performance. However, to obtain precise evaluation results, testing is **essential**. The testing process involves subsampling a dense point cloud into a sequence of voxelized point clouds, ensuring comprehensive coverage of all points. These sub-results are then predicted and collected to form a complete prediction of the entire point cloud. This approach yields  higher evaluation results compared to simply mapping/interpolating the prediction. In addition, our testing code supports TTA (test time augmentation) testing, which further enhances the stability of evaluation performance.

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-spunet-v1m1-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-spunet-v1m1-0-base.py --options save_path=exp/scannet/semseg-spunet-v1m1-0-base weight=exp/scannet/semseg-spunet-v1m1-0-base/model/model_best.pth
```

The TTA can be disabled by replace `data.test.test_cfg.aug_transform = [...]` with:

```python
data = dict(
    train = dict(...),
    val = dict(...),
    test = dict(
        ...,
        test_cfg = dict(
            ...,
            aug_transform = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]
        )
    )
)
```