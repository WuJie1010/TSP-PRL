# Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video (AAAI2020)
This repository contains the pytorch codes and trained models described in the paper "Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video" By Jie Wu, Guanbin Li, Si Liu, Liang Lin. 

## Motivation
![Motivation](https://github.com/WuJie1010/TSP-PRL/blob/master/images/introduction.png)

## Framework
![Framework](https://github.com/WuJie1010/TSP-PRL/blob/master/images/model.png)

## Requirements
- Python 2.7
- Pytorch 0.4.1
- matplotlib
- The code is for [Charades-STA](https://arxiv.org/pdf/1705.02101.pdf) dataset.

## Visual Features
Please download the features in [Features1](https://drive.google.com/drive/folders/1U1GEti3JjLfOAN0AhCb0VXqfGoKV9qMo?usp=sharing), and put it in the "Dataset/Charades" folder.

## Training and Testing Data
Please download the TrainingData in [TrainingData](https://drive.google.com/file/d/14aAJf1Wgn6wHFGdMHDgZBphKG1FbZt_y/view?usp=sharing), and put it in the "Dataset/Charades/ref_info" folder.
Please download the TestingData in [TestingData](https://drive.google.com/file/d/1mg2ru344tzL20iQNRCT2WAzfiHEjppLB/view?usp=sharing), and put it in the "Dataset/Charades/ref_info" folder.

## Pre-trained models
We provide the pre-trained model for Charades-STA dataset, which can get 24.73 on R@1, IoU0.7 and 45.30 on R@1, IoU0.5: [Models](https://drive.google.com/open?id=1lyOlcKR5PY7cN_yjZfGUypfNRTokw_He)

## Train ###
```
python train.py
```

## Validate ###
```
python val.py
```

## Test from Pre-trained Model ###
```
python test.py
```
