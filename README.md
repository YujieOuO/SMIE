# SMIE
This is an official PyTorch implementation of **"Zero-shot Skeleton-based Action Recognition 
via Mutual Information Estimation and Maximization" in ACMMM 2023**.

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
For the Semantic Features, You can download in BaiduYun link: [Semantic Feature](https://pan.baidu.com/s/1y2r15lxGF3i9aPa1ARfRiQ).

The code: smie
* [dataset]_embeddings.npy: based on label names using Sentence-Bert.
* [dataset]_clip_embeddings.npy: based on label names using CLIP.
* [dataset]_des_embeddings.npy: based on label descriptions using Sentence-Bert.

Put the semantic feautures in fold: ./data/language/

### Label Descriptions
Using [ChatGPT](https://chat.openai.com/) to expand each action label name into a complete action description.
The total label descriptions can be found in [folder](https://github.com/YujieOuO/SMIE/tree/main/descriptions).

## Different Experiment Settings
Our SMIE employs two experiment setting.
* SynSE Experiment Setting: two datasets are used, split_5 and split_12 on NTU60, and split_10 and split_24 on NTU120. The visual feature extractor is Shift-GCN. 
* Optimized Experiment Setting: three datasets are used (NTU-60, NTU-120, PKU-MMD), and each dataset have three random splits. The visual feature extractor is classical ST-GCN to minimize the impact of the feature extractor and focus on the connection model.

### SynSE Experiment Setting
To compared with the SOTA method [SynSE](https://github.com/skelemoa/synse-zsl), 
we first apply their zero-shot class splits for SynSE Experiment Setting. You can download the visual features from their repo, 
or download from our BaiduYun link: [SOTA visual features](https://pan.baidu.com/s/1Y0nTRZ19UqnXTBJeAFPXeg). Code:smie.

Example for training and testing on NTU-60 split_5 data.
```bash
# SynSE Experiment Setting
$ python procedure.py with 'train_mode="sota"'
```
You can also choose different split id of [config.py](https://github.com/YujieOuO/SMIE/blob/main/config.py) (sota compare part).  
### Optimized Experiment Setting

#### Seen and Unseen Classes Splits
For different class splits, you can change the split_id in [split.py](https://github.com/YujieOuO/SMIE/tree/main/split.py).
Then run the split.py to obtain split data for different seen and unseen classes.
```bash
# class-split
$ python split.py
```
#### Acquire the Visual Features
Refer to [Generate_Feature](https://github.com/YujieOuO/SMIE_Generate_Feature).

#### Training & Testing
Example for training and testing on NTU-60 split_1.  
You can change some settings of [config.py](https://github.com/YujieOuO/SMIE/blob/main/config.py).  
```bash
# Optimized Experiment Setting
$ python procedure.py with 'train_mode="main"'
```

## Reference
If you find our paper and repo useful, please cite our paper. Thanks!
```
@inproceedings{zhou2023zero,
  title={Zero-shot Skeleton-based Action Recognition via Mutual Information Estimation and Maximization},
  author={Zhou, Yujie and Qiang, Wenwen and Rao, Anyi and Lin, Ning and Su, Bing and Wang, Jiaqi},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5302--5310},
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
