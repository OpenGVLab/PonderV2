<div align='center'>

<h2>PonderV2: Pave the Way for 3D Foundation Model <br>with A Universal Pre-training Paradigm</h2>

[Haoyi Zhu](https://www.haoyizhu.site/)<sup>1,4*</sup>, [Honghui Yang](https://github.com/Nightmare-n)<sup>1,3*</sup>, [Xiaoyang Wu](https://xywu.me/)<sup>1,2*</sup>, [Di Huang](https://github.com/dihuangdh)<sup>1*</sup>, [Sha Zhang](https://github.com/zhangsha1024)<sup>1,4</sup>, [Xianglong He](https://scholar.google.com/citations?hl=zh-CN&user=jKFeol0AAAAJ)<sup>1</sup>,
<br>
[Tong He](http://tonghe90.github.io/)<sup>1</sup>, [Hengshuang Zhao](https://hszhao.github.io/)<sup>2</sup>, [Chunhua Shen](https://cshen.github.io/)<sup>3</sup>, [Yu Qiao](https://mmlab.siat.ac.cn/yuqiao/)<sup>1</sup>, [Wanli Ouyang](https://wlouyang.github.io/)<sup>1</sup>
 
<sup>1</sup>[Shanghai AI Lab](https://www.shlab.org.cn/), <sup>2</sup>[HKU](https://www.hku.hk/), <sup>3</sup>[ZJU](https://www.zju.edu.cn/), <sup>4</sup>[USTC](https://en.ustc.edu.cn/) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ponderv2-pave-the-way-for-3d-foundataion/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=ponderv2-pave-the-way-for-3d-foundataion)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ponderv2-pave-the-way-for-3d-foundataion/3d-semantic-segmentation-on-scannet200)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-scannet200?p=ponderv2-pave-the-way-for-3d-foundataion)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ponderv2-pave-the-way-for-3d-foundataion/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=ponderv2-pave-the-way-for-3d-foundataion)

</div>


<p align="center">
    <img src="assets/radar.png" alt="radar" width="500" />
</p>


This is the official implementation of paper "PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm". 

PonderV2 is a comprehensive 3D pre-training framework designed to facilitate the acquisition of efficient 3D representations, thereby establishing a pathway to 3D foundational models. It is a novel universal paradigm to learn point cloud representations by differentiable neural rendering, serving as a bridge between 3D and 2D worlds. 

<p align="center">
    <img src="assets/pipeline.png" alt="pipeline" width="800" />
</p>

## Highlights:
- *Oct. 2023*: **PonderV2** is released on [arXiv](https://arxiv.org/abs/2310.08586), code will be made public and supported by [Pointcept](https://github.com/Pointcept/Pointcept) soon.

## Citation
```bib
@misc{zhu2023ponderv2,
      title={PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm}, 
      author={Haoyi Zhu and Honghui Yang and Xiaoyang Wu and Di Huang and Sha Zhang and Xianglong He and Tong He and Hengshuang Zhao and Chunhua Shen and Yu Qiao and Wanli Ouyang},
      year={2023},
      eprint={2310.08586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{huang2023ponder,
  title={Ponder: Point cloud pre-training via neural rendering},
  author={Huang, Di and Peng, Sida and He, Tong and Yang, Honghui and Zhou, Xiaowei and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16089--16098},
  year={2023}
}

@misc{yang2023unipad,
      title={UniPAD: A Universal Pre-training Paradigm for Autonomous Driving}, 
      author={Honghui Yang and Sha Zhang and Di Huang and Xiaoyang Wu and Haoyi Zhu and Tong He and Shixiang Tang and Hengshuang Zhao and Qibo Qiu and Binbin Lin and Xiaofei He and Wanli Ouyang},
      year={2023},
      eprint={2310.08370},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
