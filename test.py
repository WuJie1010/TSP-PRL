" Test file "

from __future__ import print_function

import torch.nn.functional as F
import os
import argparse
from utils import *
from dataloader_charades import Charades_Test_dataset
from model import HRL
import random
import matplotlib
import json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 4.0)

parser = argparse.ArgumentParser(description='Video Grounding of PyTorch')
parser.add_argument('--model', type=str, default='TSP_PRL', help='model type')
parser.add_argument('--dataset', type=str, default='Charades', help='dataset type')
parser.add_argument('--num_steps', type=int, default=20, help='number of forward steps in A2C (default: 10)')

opt = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

path = os.path.join(opt.dataset + '_' + opt.model)

test_dataset = Charades_Test_dataset()

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=4)

num_test_batches = int(len(test_dataset))
print ("num_test_batches:", num_test_batches)

# Model
if opt.model == 'TSP_PRL': #  JJ_batch_stop_SmoothL1Loss_Previou_IoU
    net = HRL().cuda()


def determine_scale_range(action_index, current_offset, num_units):

    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)

    abnormal_done = False
    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])

    current_offset_center = (current_offset_start + current_offset_end) /2
    length = current_offset_end - current_offset_start

    num_units_index = int(num_units)

    if action_index == 0: # expand 1.2
        current_offset_start = current_offset_center - int(0.6*length)
        current_offset_end = current_offset_center + int(0.6*length)
    elif action_index == 1: # expand 1.5
        current_offset_start = current_offset_center - int(0.75*length)
        current_offset_end = current_offset_center + int(0.75*length)
    elif action_index == 2: # reduce 1.2
        current_offset_start = current_offset_center - int(0.417*length)
        current_offset_end = current_offset_center + int(0.417*length)
    elif action_index == 3: # reduce 1.5
        current_offset_start = current_offset_center - int(0.333*length)
        current_offset_end = current_offset_center + int(0.333*length)
    else:
        abnormal_done = True #stop

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end >= num_units_index:
        current_offset_end = num_units_index -1
        if current_offset_start > num_units_index:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    if not abnormal_done:

        current_offset_start_norm = current_offset_start / float(num_units_index - 1)
        current_offset_end_norm = current_offset_end / float(num_units_index - 1)

        update_offset_norm[0] = current_offset_start_norm
        update_offset_norm[1] = current_offset_end_norm

        update_offset[0] = current_offset_start
        update_offset[1] = current_offset_end
        update_offset = torch.from_numpy(update_offset)
        update_offset = update_offset.unsqueeze(0).cuda()

        update_offset_norm = torch.from_numpy(update_offset_norm)
        update_offset_norm = update_offset_norm.unsqueeze(0).cuda()

    else:
        update_offset = current_offset

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_left_move_range(action_index, current_offset, ten_unit, num_units):

    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)
    abnormal_done = False

    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])

    ten_unit_index = int(ten_unit)
    num_units_index = int(num_units)

    if action_index == 0:
        current_offset_start = current_offset_start - ten_unit_index
        current_offset_end = current_offset_end - ten_unit_index
    elif action_index == 1:
        current_offset_start = current_offset_start - ten_unit_index
    elif action_index == 2:
        current_offset_end = current_offset_end - ten_unit_index
    else:
        abnormal_done = True #stop

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end >= num_units_index:
        current_offset_end = num_units_index -1
        if current_offset_start > num_units_index:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    if not abnormal_done:
        current_offset_start_norm = current_offset_start / float(num_units_index - 1)
        current_offset_end_norm = current_offset_end / float(num_units_index - 1)

        update_offset_norm[0] = current_offset_start_norm
        update_offset_norm[1] = current_offset_end_norm

        update_offset[0] = current_offset_start
        update_offset[1] = current_offset_end

        update_offset = torch.from_numpy(update_offset)
        update_offset = update_offset.unsqueeze(0).cuda()

        update_offset_norm = torch.from_numpy(update_offset_norm)
        update_offset_norm = update_offset_norm.unsqueeze(0).cuda()

    else:
        update_offset = current_offset


    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_right_move_range(action_index, current_offset, ten_unit, num_units):

    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)
    abnormal_done = False

    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])

    ten_unit_index = int(ten_unit)
    num_units_index = int(num_units)

    if action_index == 0:
        current_offset_start = current_offset_start + ten_unit_index
        current_offset_end = current_offset_end + ten_unit_index
    elif action_index == 1:
        current_offset_start = current_offset_start + ten_unit_index
    elif action_index == 2:
        current_offset_end = current_offset_end + ten_unit_index
    else:
        abnormal_done = True #stop

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end >= num_units_index:
        current_offset_end = num_units_index -1
        if current_offset_start > num_units_index:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    if not abnormal_done:
        current_offset_start_norm = current_offset_start / float(num_units_index - 1)
        current_offset_end_norm = current_offset_end / float(num_units_index - 1)

        update_offset_norm[0] = current_offset_start_norm
        update_offset_norm[1] = current_offset_end_norm

        update_offset[0] = current_offset_start
        update_offset[1] = current_offset_end

        update_offset = torch.from_numpy(update_offset)
        update_offset = update_offset.unsqueeze(0).cuda()

        update_offset_norm = torch.from_numpy(update_offset_norm)
        update_offset_norm = update_offset_norm.unsqueeze(0).cuda()

    else:
        update_offset = current_offset


    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_left_offset_range(action_index, current_offset, num_units):

    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)

    abnormal_done = False

    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])

    num_units_index = int(num_units)

    if action_index == 0:
        current_offset_start = current_offset_start - 1
        current_offset_end = current_offset_end - 1
    elif action_index == 1:
        current_offset_start = current_offset_start - 1
    elif action_index == 2:
        current_offset_end = current_offset_end - 1
    else:
        abnormal_done = True #stop

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end >= num_units_index:
        current_offset_end = num_units_index -1
        if current_offset_start > num_units_index:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    if not abnormal_done:

        current_offset_start_norm = current_offset_start / float(num_units_index - 1)
        current_offset_end_norm = current_offset_end / float(num_units_index - 1)

        update_offset_norm[0] = current_offset_start_norm
        update_offset_norm[1] = current_offset_end_norm

        update_offset[0] = current_offset_start
        update_offset[1] = current_offset_end

        update_offset = torch.from_numpy(update_offset)
        update_offset = update_offset.unsqueeze(0).cuda()

        update_offset_norm = torch.from_numpy(update_offset_norm)
        update_offset_norm = update_offset_norm.unsqueeze(0).cuda()
    else:
        update_offset = current_offset

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_right_offset_range(action_index, current_offset, num_units):

    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)

    abnormal_done = False

    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])

    num_units_index = int(num_units)

    if action_index == 0:
        current_offset_start = current_offset_start + 1
        current_offset_end = current_offset_end + 1
    elif action_index == 1:
        current_offset_start = current_offset_start + 1
    elif action_index == 2:
        current_offset_end = current_offset_end + 1
    else:
        abnormal_done = True #stop

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end >= num_units_index:
        current_offset_end = num_units_index -1
        if current_offset_start > num_units_index:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    if not abnormal_done:

        current_offset_start_norm = current_offset_start / float(num_units_index - 1)
        current_offset_end_norm = current_offset_end / float(num_units_index - 1)

        update_offset_norm[0] = current_offset_start_norm
        update_offset_norm[1] = current_offset_end_norm

        update_offset[0] = current_offset_start
        update_offset[1] = current_offset_end

        update_offset = torch.from_numpy(update_offset)
        update_offset = update_offset.unsqueeze(0).cuda()

        update_offset_norm = torch.from_numpy(update_offset_norm)
        update_offset_norm = update_offset_norm.unsqueeze(0).cuda()
    else:
        update_offset = current_offset

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done


def test():

    checkpoint = torch.load(os.path.join(path, 'best_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.eval()

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    all_correct_num_1 = [0.0] * 5
    all_correct_num_20 = [0.0] * 5
    all_correct_num_30 = [0.0] * 5
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names)
    idx = 0
    iou_sum_stop_10_all = 0
    iou_sum_stop_20_all = 0
    iou_sum_stop_30_all = 0

    for batch_idx, (
    movie_clip_sentences, global_feature, original_feats, initial_feature, initial_offset, initial_offset_norm,
    ten_unit, num_units) in enumerate(testloader):
        global_feature, original_feats, initial_feature, initial_offset, initial_offset_norm, ten_unit, num_units = \
            global_feature.cuda(), original_feats.cuda(), initial_feature.cuda(), initial_offset.cuda(), initial_offset_norm.cuda(), ten_unit.cuda(), num_units.cuda()
        idx += 1
        print("%d/%d" % (idx, all_number))
        print("sentences: " + str(len(movie_clip_sentences)))

        sentence_image_reg_mat_10 = np.zeros([len(movie_clip_sentences), 2])
        sentence_image_reg_mat_20 = np.zeros([len(movie_clip_sentences), 2])
        sentence_image_reg_mat_30 = np.zeros([len(movie_clip_sentences), 2])

        for k in range(len(movie_clip_sentences)):

            sent_vec = movie_clip_sentences[k][1][0]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]]).cuda()  # 1,4800
            # sent_vec = torch.from_numpy(sent_vec).cuda()

            # network forward

            best_iou_out = -1e8
            best_iou_out_20 = -1e8
            best_iou_out_30 = -1e8

            for step in range(opt.num_steps):
                if step == 0:
                    hidden_state = torch.zeros(1, 1024).cuda()
                    current_feature = initial_feature
                    current_offset = initial_offset
                    current_offset_norm = initial_offset_norm
                    # print(initial_offset[0][0], initial_offset[0][1])
                    current_offset_start = initial_offset[0][0]
                    current_offset_end = initial_offset[0][1]

                hidden_state, global_policy, global_value, scale_policy, scale_value, left_move_policy, left_move_value, right_move_policy, right_move_value, \
                left_offset_policy, left_offset_value, right_offset_policy, right_offset_value, iou_out = net(
                    global_feature, current_feature, sent_vec, current_offset_norm, hidden_state)

                if step < 10:
                    if iou_out > best_iou_out:
                        best_iou_out = iou_out
                        best_step = step+1
                        best_current_offset_start = current_offset_start
                        best_current_offset_end = current_offset_end

                if step < 20:
                    if iou_out > best_iou_out_20:
                        best_iou_out_20 = iou_out
                        best_step = step+1
                        best_current_offset_start_20 = current_offset_start
                        best_current_offset_end_20 = current_offset_end

                if step < 30:
                    if iou_out > best_iou_out_30:
                        best_iou_out_30 = iou_out
                        best_step = step+1
                        best_current_offset_start_30 = current_offset_start
                        best_current_offset_end_30 = current_offset_end

                global_policy_prob = F.softmax(global_policy, dim=1)
                global_action = global_policy_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]
                # print(global_policy_prob)
                # print("Step: %d, action: %d" %((step+1), global_action))

                if global_action == 0:
                    scale_policy_prob = F.softmax(scale_policy[0], dim=0)
                    scale_policy_action = scale_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()

                    current_offset_start, current_offset_end, current_offset, current_offset_norm, abnormal_done = determine_scale_range(
                        scale_policy_action, current_offset, num_units)

                elif global_action ==1:
                    left_move_policy_prob = F.softmax(left_move_policy[0], dim=0)
                    left_move_policy_action = left_move_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()

                    current_offset_start, current_offset_end, current_offset, current_offset_norm, abnormal_done = determine_left_move_range(
                        left_move_policy_action, current_offset, ten_unit, num_units)

                elif global_action ==2:
                    right_move_policy_prob = F.softmax(right_move_policy[0], dim=0)
                    right_move_policy_action = right_move_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()

                    current_offset_start, current_offset_end, current_offset, current_offset_norm, abnormal_done = determine_right_move_range(
                        right_move_policy_action, current_offset, ten_unit, num_units)

                elif global_action == 3:
                    left_offset_policy_prob = F.softmax(left_offset_policy[0], dim=0)
                    left_offset_policy_action = left_offset_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()

                    current_offset_start, current_offset_end, current_offset, current_offset_norm, abnormal_done = determine_left_offset_range(
                        left_offset_policy_action, current_offset, num_units)

                else:
                    right_offset_policy_prob = F.softmax(right_offset_policy[0], dim=0)
                    right_offset_policy_action = right_offset_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()

                    current_offset_start, current_offset_end, current_offset, current_offset_norm, abnormal_done = determine_right_offset_range(
                        right_offset_policy_action, current_offset, num_units)


                if abnormal_done == False:
                    current_feature = original_feats[0][(current_offset_start):(current_offset_end + 1)]

                    feature_length = len(current_feature)
                    index = choose_four_frame(feature_length)
                    initial_feature_1 = current_feature[index[0]]
                    initial_feature_2 = current_feature[index[1]]
                    initial_feature_3 = current_feature[index[2]]
                    initial_feature_4 = current_feature[index[3]]
                    initial_feature_concate = torch.cat((initial_feature_1, initial_feature_2, initial_feature_3, initial_feature_4), 0)
                    current_feature = initial_feature_concate
                    current_feature = current_feature.unsqueeze(0).cuda()

                if abnormal_done == True:
                    break
            # print("emit stop at step: %d with %f " % ((best_step), best_iou_out))
            sentence_image_reg_mat_10[k, 0] = best_current_offset_start * 16 +1
            sentence_image_reg_mat_10[k, 1] = best_current_offset_end * 16 +1

            sentence_image_reg_mat_20[k, 0] = best_current_offset_start_20 * 16 +1
            sentence_image_reg_mat_20[k, 1] = best_current_offset_end_20 * 16 + 1

            sentence_image_reg_mat_30[k, 0] = best_current_offset_start_30 * 16 +1
            sentence_image_reg_mat_30[k, 1] = best_current_offset_end_30 * 16 + 1


        sclips = [b[0][0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_1, iou_sum_10 = compute_IoU_recall_top_n_forreg_rl_iou_batch(1, IoU, sentence_image_reg_mat_10, sclips)
            correct_num_20, iou_sum_20 = compute_IoU_recall_top_n_forreg_rl_iou_batch(1, IoU, sentence_image_reg_mat_20, sclips)
            correct_num_30, iou_sum_30 = compute_IoU_recall_top_n_forreg_rl_iou_batch(1, IoU, sentence_image_reg_mat_30, sclips)
            # print(" IoU=" + str(IoU) + ", R@1: " + str(correct_num_1 / len(sclips)))

            all_correct_num_1[k] += correct_num_1
            all_correct_num_20[k] += correct_num_20
            all_correct_num_30[k] += correct_num_30

        iou_sum_stop_10_all += iou_sum_10
        iou_sum_stop_20_all += iou_sum_20
        iou_sum_stop_30_all += iou_sum_30

        all_retrievd += len(sclips)

        print("Current R1_IOU7_10", all_correct_num_1[3] / all_retrievd)
        print("Current R1_IOU5_10", all_correct_num_1[2] / all_retrievd)

        print("Current R1_IOU7_20", all_correct_num_20[3] / all_retrievd)
        print("Current R1_IOU5_20", all_correct_num_20[2] / all_retrievd)

        print("Current R1_IOU7_30", all_correct_num_30[3] / all_retrievd)
        print("Current R1_IOU5_30", all_correct_num_30[2] / all_retrievd)

    for k in range(len(IoU_thresh)):
        print(" 10: IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
        print(" 20: IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_20[k] / all_retrievd))
        print(" 30: IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_30[k] / all_retrievd))

    print("Current MIOU_10", iou_sum_stop_10_all / all_retrievd)
    print("Current MIOU_20", iou_sum_stop_20_all / all_retrievd)
    print("Current MIOU_30", iou_sum_stop_30_all / all_retrievd)

if __name__ == '__main__':
    test()

