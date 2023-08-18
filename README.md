# SMIE
This is an official PyTorch implementation of **"Zero-shot Skeleton-based Action Recognition 
via Mutual Information Estimation and Maximization" in ACMMM 2023**.
(Code comming soon)

[[Paper]](https://arxiv.org/abs/2308.03950)
# Framework
![SMIE](https://github.com/YujieOuO/SMIE/blob/main/images/pipeline.png)

## Requirements
![python = 3.7](https://img.shields.io/badge/python-3.7.13-green)
![torch = 1.11.0+cu113](https://img.shields.io/badge/torch-1.11.0%2Bcu113-yellowgreen)

## Installation
```bash
# Install the python libraries
$ cd SMIE
$ pip install -r requirements.txt
```

## Data Preparation
We apply the same dataset processing as [AimCLR](https://github.com/Levigty/AimCLR).  
You can also download the skeleton data in BaiduYun link:
* [NTU-RGB-D 60](https://pan.baidu.com/s/1ukBF5aI8QawRriJbmsrv5Q).
* [NTU-RGB-D 120](https://pan.baidu.com/s/1AG_516WHitv1LBh1NNrvVg).
* [PKU-MMD](https://pan.baidu.com/s/168uXCgrKdh7esqatGwfEfg).
  
The code: pstl

### Semantic Features
For the Semantic Features, You can download in BaiduYun link: [Semantic Feature](https://pan.baidu.com/s/1y2r15lxGF3i9aPa1ARfRiQ)

The code: smie
* [dataset]_embeddings.npy: based on label names using Sentence-Bert.
* [dataset]_clip_embeddings: based on label names using CLIP.
* [dataset]_des_embeddings: based on label descriptions using Sentence-Bert.
* 
### Label descriptions
Using [ChatGPT](https://chat.openai.com/) to expand each action label name into a complete action description.
The total label description can be found in folder: 

## Seen and Unseen Classes Splits

## Visual Feature Preparation

## Reference
If you find our paper and repo useful, please cite our paper. Thanks!
```
@article{zhou2023zero,
  title={Zero-shot Skeleton-based Action Recognition via Mutual Information Estimation and Maximization},
  author={Zhou, Yujie and Qiang, Wenwen and Rao, Anyi and Lin, Ning and Su, Bing and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2308.03950},
  year={2023}
}
```

## Acknowledgement
* The codebase is from [MS2L](https://github.com/LanglandsLin/MS2L).
* The visual feature is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).
* The semantic feature is based on [Sentence-Bert](https://github.com/UKPLab/sentence-transformers).
* The baseline methods are from [SynSE](https://github.com/skelemoa/synse-zsl).

## Licence
This project is licensed under the terms of the MIT license.

## Contact
For any questions, feel free to contact: yujieouo@sjtu.edu.cn
