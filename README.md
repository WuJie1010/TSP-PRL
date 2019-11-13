# Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video (AAAI2020)
This repository contains the pytorch codes and trained models described in the paper "Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video" By Jie Wu, Guanbin Li, Si Liu, Liang Lin. 


![Motivation](https://raw.githubusercontent.com/WuJie1010/TSP-PRL/master/images/introduction.png)
![Framework](https://raw.githubusercontent.com/WuJie1010/TSP-PRL/master/images/model.png)

## Requirements
- Python 2.7
- Pytorch 0.4.1
- matplotlib
- The code is for [Charades-STA](https://arxiv.org/pdf/1705.02101.pdf) dataset.

## Features
Please download the features in [Features](https://drive.google.com/drive/folders/1U1GEti3JjLfOAN0AhCb0VXqfGoKV9qMo?usp=sharing), and put it in the "Dataset/Charades" folder.

## Pre-trained models
We provide the pre-trained model for Charades-STA dataset, which can get 24.73 on R@1, IoU0.7 and 45.30 on R@1, IoU0.5: [Models]()

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
