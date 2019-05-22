import numpy as np
from six.moves import xrange
import time
import pickle
import operator
import torch

def choose_four_frame(N):

    # if N < 4:
    idx = []

    if N ==1:
        idx = [0,0,0,0]
    elif N ==2:
        idx = [0,0,1,1]
    elif N == 3:
        idx = [0, 1, 1, 2]
    elif N == 4:
        idx = [0, 1, 2, 3]

    else:
        five_unit  = N/float(5)
        for i in range(4):
            idx.append(int(np.floor((i+1)*five_unit)))

    return idx


def calculate_root_reward_batch(Previou_IoU, current_IoU, global_all_IoU):
    batch_size = len(current_IoU)
    reward = torch.zeros(batch_size)
    global_IoU = torch.max(global_all_IoU, dim=1)[0]

    for i in range(batch_size):

        if current_IoU[i] == global_IoU[i]:
            reward[i] = 1 + current_IoU[i] - Previou_IoU[i]
        else:
            reward[i] = current_IoU[i] - global_IoU[i] + current_IoU[i] - Previou_IoU[i]  #weight?
            # current_IoU[i] - global_IoU[i]: global reward, choose best one
            # current_IoU[i] - Previou_IoU[i]: local reward, fast to detect
        # print(Previou_IoU[i], current_IoU[i], global_IoU[i], reward[i])
    return reward

def calculate_leaf_reward_batch(Previou_IoU, current_IoU, t):
    batch_size = len(Previou_IoU)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        if current_IoU[i] > Previou_IoU[i] and Previou_IoU[i]>=0:
            if current_IoU[i] > 0.5:
                reward[i] = 1 + current_IoU[i] # encourage refine
            else:
                reward[i] = 1
        elif current_IoU[i] <= Previou_IoU[i] and current_IoU[i]>=0:
            reward[i] = -0.1 #should determine
        else:
            reward[i] = -1

    return reward

def calculate_global_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    global_units = len(i0[0])
    # print("global_units", global_units)
    iou_batch = torch.zeros(batch_size, global_units)


    for i in range(batch_size):
        for j in range(global_units):
            union = (min(i0[i][j][0], i1[i][0]), max(i0[i][j][1], i1[i][1]))
            inter = (max(i0[i][j][0], i1[i][0]), min(i0[i][j][1], i1[i][1]))
            # if inter[1] < inter[0]:
            #     iou = 0
            # else:
            iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
            iou_batch[i][j] = iou
    return iou_batch

def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        # if inter[1] < inter[0]:
        #     iou = 0
        # else:
        iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
        iou_batch[i] = iou
    return iou_batch

def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou


def compute_IoU_recall_top_n_forreg_rl_batch_iou(top_n, iou_thresh, sentence_image_reg_mat, sclips):
    correct_num = 0.0
    iou_sum = 0
    for k in range(sentence_image_reg_mat.shape[0]):
        gt = sclips[k]
        # print(gt)
        gt_start = gt[0]
        gt_end = gt[1]

        pred_start = sentence_image_reg_mat[k, 0]
        pred_end = sentence_image_reg_mat[k, 1]
        iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
        if iou < 0:
            iou = 0
        iou_sum += iou
        if iou>=iou_thresh:
            correct_num+=1

    return correct_num, iou_sum

'''
compute recall at certain IoU
'''
def compute_IoU_recall_top_n_forreg_rl(top_n, iou_thresh, sentence_image_reg_mat, sclips):
    correct_num = 0.0
    for k in range(sentence_image_reg_mat.shape[0]):
        gt = sclips[k]
        # print(gt)
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])

        pred_start = sentence_image_reg_mat[k, 0]
        pred_end = sentence_image_reg_mat[k, 1]
        iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
        if iou>=iou_thresh:
            correct_num+=1

    return correct_num

def compute_IoU_recall_top_n_forreg_rl_iou_batch(top_n, iou_thresh, sentence_image_reg_mat, sclips):

    correct_num = 0.0
    iou_num = 0.0
    for k in range(sentence_image_reg_mat.shape[0]):
        gt = sclips[k]
        # print(gt)
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])

        pred_start = sentence_image_reg_mat[k, 0]
        pred_end = sentence_image_reg_mat[k, 1]
        iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
        if iou < 0:
            iou = 0
        iou_num += iou

        if iou>=iou_thresh:
            correct_num+=1

    return correct_num, iou_num
