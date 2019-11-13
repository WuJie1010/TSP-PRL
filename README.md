# Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video (AAAI2020)
By Jie Wu, Guanbin Li, Si Liu, Liang Lin. This repository contains the pytorch codes and trained models described in the paper "Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video".

## Requirements
Python2.7, pytorch 0.4.1

## Features
Please download the features in [link](https://drive.google.com/drive/folders/1U1GEti3JjLfOAN0AhCb0VXqfGoKV9qMo?usp=sharing), and put it in the "Dataset/Charades" folder.

## Pre-trained models
We provide the pre-trained model for Charades-STA dataset, which can get 24.73 on R@1, IoU0.7 and 45.30 on R@1, IoU0.5: [link]()

## Train ###
- python train.py

## Validate ###
- python val.py

## Test from Pre-trained Model ###
- python test.py
