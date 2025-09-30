## Model Zoo

*For outdoor models, please check out [UniPAD](https://github.com/Nightmare-n/UniPAD)!*

### Pretraining

Model | Backbone | HuggingFace | Google Drive | Baidu Pan | Config
----- | ----- | -----|  ----- | ----- | -----
[PonderIndoor-v2](../ponder/models/ponder/ponder_indoor_base.py) | [SpUNet-v1m3](../ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py) | [ckpt](https://huggingface.co/HaoyiZhu/PonderV2/blob/main/checkpoints/ponderv2-ppt-pretrain-scannet-s3dis-structured3d.pth) | [ckpt](https://drive.google.com/file/d/1oFHhr6YPZwgUCtn0y9M6zAHL7XNkmpVM/view?usp=sharing) | [ckpt](https://pan.baidu.com/s/1mF5BdjvS2DDjFrqntm-kYA?pwd=soin) | [cfg](../configs/scannet/pretrain-ponder-ppt-v1m1-0-sc-s3-st-spunet.py)

### Finetuning
#### Indoor Semantic Segmentation

Benchmark | Model | Backbone | Val mIoU | Test mIoU | HuggingFace | Google Drive | Baidu Pan | Config
----- | ----- | ----- | ----- | ----- | ----- |  ----- | ----- | -----
ScanNet | [PPT-v1m1](../ponder/models/point_prompt_training/point_prompt_training_v1m1_language_guided.py) | [SpUNet-v1m3](../ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py) | 77.0 | 78.5 | [ckpt](https://huggingface.co/HaoyiZhu/PonderV2/blob/main/checkpoints/ponderv2-ppt-ft-semseg-scannet.pth) | [ckpt](https://drive.google.com/file/d/16RhUSJDxtsS7Z_FeRJwk0L3q7Dt-LN1A/view?usp=sharing) | [ckpt](https://pan.baidu.com/s/1l0_k0h9fvnI38By6bSdijQ?pwd=wks7) | [cfg](../configs/scannet/semseg-ppt-v1m1-0-sc-s3-st-spunet-lovasz-ft.py)
ScanNet200 | [PPT-v1m1](../ponder/models/point_prompt_training/point_prompt_training_v1m1_language_guided.py) | [SpUNet-v1m3](../ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py) | 32.3 | 34.6 | [ckpt](https://huggingface.co/HaoyiZhu/PonderV2/blob/main/checkpoints/ponderv2-ppt-ft-semseg-scannet200.pth) | [ckpt](https://drive.google.com/file/d/1d_we6SsNJLDeRc1LepZwcOCgOGJy11rr/view?usp=sharing) | [ckpt](https://pan.baidu.com/s/1fvH5wA60wl2In0BaUMuFOw?pwd=3ron) | [cfg](../configs/scannet200/semseg-ppt-v1m1-0-spunet-lovasz-ft.py)
S3DIS (Area5) | [PPT-v1m1](../ponder/models/point_prompt_training/point_prompt_training_v1m1_language_guided.py) | [SpUNet-v1m3](../ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py) | 73.2 | 79.1 | [ckpt](https://huggingface.co/HaoyiZhu/PonderV2/blob/main/checkpoints/ponderv2-ppt-ft-semseg-s3dis.pth) | [ckpt](https://drive.google.com/file/d/1GZgfxWJC9hNEHKV30t3PuIZMmTXngHXg/view?usp=sharing) | [ckpt](https://pan.baidu.com/s/1KJ-PwvROofcGeTzkKdsXCQ?pwd=bbaa) | [cfg](../configs/s3dis/semseg-ppt-v1m1-0-s3-sc-st-spunet-lovasz-ft.py)

#### Indoor Instance Segmentation
Benchmark | Model | Backbone | mAP@25 | mAP@50 | mAP | HuggingFace | Google Drive | Baidu Pan | Config
----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | -----
ScanNet | [PPT-v1m1](../ponder/models/point_prompt_training/point_prompt_training_v1m1_language_guided.py) | [SpUNet-v1m3](../ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py) | 77.0 | 62.6 | 40.9 | [ckpt](https://huggingface.co/HaoyiZhu/PonderV2/blob/main/checkpoints/ponderv2-ppt-ft-insseg-scannet.pth) | [ckpt](https://drive.google.com/file/d/15tjsGY6bgZiQSXJel7yywzdYBDrpystk/view?usp=sharing) | [ckpt](https://pan.baidu.com/s/10BifGbWQ6CW_FcAnaw-XQg?pwd=jmd9) | [cfg](../configs/scannet/insseg-ppt-v1m1-0-pointgroup-spunet-ft.py)
ScanNet200 | [PPT-v1m1](../ponder/models/point_prompt_training/point_prompt_training_v1m1_language_guided.py) | [SpUNet-v1m3](../ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py) | 37.6| 30.5 | 20.1 | [ckpt](https://huggingface.co/HaoyiZhu/PonderV2/blob/main/checkpoints/ponderv2-ppt-ft-insseg-scannet200.pth) | [ckpt](https://drive.google.com/file/d/1MVC2xSgXqbFDzIlni1KyPT28thbAIwm6/view?usp=sharing) | [ckpt](https://pan.baidu.com/s/1MbCPJEWbgOOmoB3riCEfcg?pwd=6pm0) | [cfg](../configs/scannet200/insseg-ppt-v1m1-0-pointgroup-spunet-ft.py)

