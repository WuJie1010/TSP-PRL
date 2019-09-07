" Training file for Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video  "

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
from utils import *
from dataloader_charades import Charades_Train_dataset
from model import HRL
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 4.0)

parser = argparse.ArgumentParser(description='Video Grounding of PyTorch')
parser.add_argument('--model', type=str, default='TSP_PRL', help='model type')
parser.add_argument('--dataset', type=str, default='Charades', help='dataset type')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--num_steps', type=int, default=20, help='number of forward steps in A2C (default: 10)')
parser.add_argument('--gamma', type=float, default=0.4,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.1,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--switch_iteration', type=int, default=200,
                    help='switch training')

opt = parser.parse_args()
regression_loss_func = nn.BCEWithLogitsLoss()# nn.BCEWithLogitsLoss()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(0)

path = os.path.join(opt.dataset + '_' + opt.model)

if not os.path.exists(path):
        os.makedirs(path)

train_dataset = Charades_Train_dataset()

num_train_batches = int(len(train_dataset)/opt.batch_size)
print ("num_train_batches:", num_train_batches)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4)

# Model
if opt.model == 'TSP_PRL':
    net = HRL().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)


def determine_scale_range(action_index, current_offset, num_units):
    update_offset = torch.zeros(2)
    update_offset_norm = torch.zeros(2)
    abnormal_done = 1
    current_offset_start = int(current_offset[0])
    current_offset_end = int(current_offset[1])
    current_offset_center = (current_offset_start + current_offset_end) /2
    length = current_offset_end - current_offset_start
    num_units_index = int(num_units)

    if current_offset_end < 0 or current_offset_start > num_units_index or current_offset_end <= current_offset_start:
        abnormal_done = 0
    else:

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
            abnormal_done = 0 #stop

        if current_offset_start < 0:
            current_offset_start = 0
            if current_offset_end < 0:
                abnormal_done = 0

        if current_offset_end >= num_units_index:
            current_offset_end = num_units_index -1
            if current_offset_start >= num_units_index:
                abnormal_done = 0

        if current_offset_end <= current_offset_start:
            abnormal_done = 0

    current_offset_start_norm = current_offset_start / float(num_units_index - 1)
    current_offset_end_norm = current_offset_end / float(num_units_index - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done


def determine_left_move_range(action_index, current_offset, ten_unit, num_units):

    update_offset = torch.zeros(2)
    update_offset_norm = torch.zeros(2)

    abnormal_done = 1

    current_offset_start = int(current_offset[0])
    current_offset_end = int(current_offset[1])

    ten_unit_index = int(ten_unit)
    num_units_index = int(num_units)

    if current_offset_end < 0 or current_offset_start > num_units_index or current_offset_end <= current_offset_start:
        abnormal_done = 0
    else:

        if action_index == 0:
            current_offset_start = current_offset_start - ten_unit_index
            current_offset_end = current_offset_end - ten_unit_index
        elif action_index == 1:
            current_offset_start = current_offset_start - ten_unit_index
        elif action_index == 2:
            current_offset_end = current_offset_end - ten_unit_index
        else:
            abnormal_done = 0 #stop

        if current_offset_start < 0:
            current_offset_start = 0
            if current_offset_end < 0:
                abnormal_done = 0

        if current_offset_end >= num_units_index:
            current_offset_end = num_units_index -1
            if current_offset_start > num_units_index:
                abnormal_done = 0

        if current_offset_end <= current_offset_start:
            abnormal_done = 0

    current_offset_start_norm = current_offset_start / float(num_units_index - 1)
    current_offset_end_norm = current_offset_end / float(num_units_index - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_right_move_range(action_index, current_offset, ten_unit, num_units):

    update_offset = torch.zeros(2)
    update_offset_norm = torch.zeros(2)

    abnormal_done = 1

    current_offset_start = int(current_offset[0])
    current_offset_end = int(current_offset[1])

    ten_unit_index = int(ten_unit)
    num_units_index = int(num_units)

    if current_offset_end < 0 or current_offset_start > num_units_index or current_offset_end <= current_offset_start:
        abnormal_done = 0
    else:

        if action_index == 0:
            current_offset_start = current_offset_start + ten_unit_index
            current_offset_end = current_offset_end + ten_unit_index
        elif action_index == 1:
            current_offset_start = current_offset_start + ten_unit_index
        elif action_index == 2:
            current_offset_end = current_offset_end + ten_unit_index
        else:
            abnormal_done = 0 #stop

        if current_offset_start < 0:
            current_offset_start = 0
            if current_offset_end < 0:
                abnormal_done = 0

        if current_offset_end >= num_units_index:
            current_offset_end = num_units_index -1
            if current_offset_start > num_units_index:
                abnormal_done = 0

        if current_offset_end <= current_offset_start:
            abnormal_done = 0

    current_offset_start_norm = current_offset_start / float(num_units_index - 1)
    current_offset_end_norm = current_offset_end / float(num_units_index - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_left_offset_range(action_index, current_offset, num_units):

    update_offset = torch.zeros(2)
    update_offset_norm = torch.zeros(2)

    abnormal_done = 1

    current_offset_start = int(current_offset[0])
    current_offset_end = int(current_offset[1])

    num_units_index = int(num_units)

    if current_offset_end < 0 or current_offset_start > num_units_index or current_offset_end <= current_offset_start:
        abnormal_done = 0
    else:

        if action_index == 0:
            current_offset_start = current_offset_start - 1
            current_offset_end = current_offset_end - 1
        elif action_index == 1:
            current_offset_start = current_offset_start - 1
        elif action_index == 2:
            current_offset_end = current_offset_end - 1
        else:
            abnormal_done = 0 #stop

        if current_offset_start < 0:
            current_offset_start = 0
            if current_offset_end < 0:
                abnormal_done = 0

        if current_offset_end >= num_units_index:
            current_offset_end = num_units_index -1
            if current_offset_start > num_units_index:
                abnormal_done = 0

        if current_offset_end <= current_offset_start:
            abnormal_done = 0

    current_offset_start_norm = current_offset_start / float(num_units_index - 1)
    current_offset_end_norm = current_offset_end / float(num_units_index - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done

def determine_right_offset_range(action_index, current_offset, num_units):

    update_offset = torch.zeros(2)
    update_offset_norm = torch.zeros(2)

    abnormal_done = 1

    current_offset_start = int(current_offset[0])
    current_offset_end = int(current_offset[1])

    num_units_index = int(num_units)

    if current_offset_end < 0 or current_offset_start > num_units_index or current_offset_end <= current_offset_start:
        abnormal_done = 0
    else:

        if action_index == 0:
            current_offset_start = current_offset_start + 1
            current_offset_end = current_offset_end + 1
        elif action_index == 1:
            current_offset_start = current_offset_start + 1
        elif action_index == 2:
            current_offset_end = current_offset_end + 1
        else:
            abnormal_done = 0 #stop

        if current_offset_start < 0:
            current_offset_start = 0
            if current_offset_end < 0:
                abnormal_done = 0

        if current_offset_end >= num_units_index:
            current_offset_end = num_units_index -1
            if current_offset_start > num_units_index:
                abnormal_done = 0

        if current_offset_end <= current_offset_start:
            abnormal_done = 0

    current_offset_start_norm = current_offset_start / float(num_units_index - 1)
    current_offset_end_norm = current_offset_end / float(num_units_index - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    return current_offset_start, current_offset_end, update_offset, update_offset_norm, abnormal_done


def freeze_net(net, global_flag):

    if global_flag == True:
        # don't compute the gradient of local polict network
        ct = 0
        for child in net.children():
            ct +=1
            if ct > 8 and ct < 19:
                print(child)
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    else:
        # don't compute the gradient of global polict network
        ct = 0
        for child in net.children():
            ct +=1
            if ct == 7 or ct == 8:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

# Training
def train(start_epoch, total_epoch):
    iteration = 0
    global_flag = True

    net.train()

    global_policy_loss_epoch = []
    global_value_loss_epoch = []
    local_policy_loss_epoch = []
    local_value_loss_epoch = []

    iou_loss_epoch = []
    total_rewards_epoch = []
    global_total_rewards_epoch = []

    for epoch in range(start_epoch, total_epoch):
        start = time.time()

        for batch_idx, (global_feature, original_feats, initial_feature, sentence, offset_norm, initial_offset, initial_offset_norm, ten_unit, num_units) in enumerate(trainloader):

            global_feature, original_feats, initial_feature, sentence, offset_norm, initial_offset, initial_offset_norm, ten_unit, num_units = global_feature.cuda(), \
             original_feats.cuda(), initial_feature.cuda(), sentence.cuda(), offset_norm.cuda(), initial_offset.cuda(), initial_offset_norm.cuda(), ten_unit.cuda(), num_units.cuda()

            if iteration % opt.switch_iteration == 0:
                global_flag = not global_flag
                print("Switch state to: ",global_flag)
                freeze_net(net, global_flag)

            batch_size = len(global_feature)
            global_policy_entropies = torch.zeros(opt.num_steps, batch_size)
            local_policy_entropies = torch.zeros(opt.num_steps, batch_size)
            global_policy_values = torch.zeros(opt.num_steps, batch_size)
            global_policy_log_probs = torch.zeros(opt.num_steps, batch_size)
            global_policy_rewards = torch.zeros(opt.num_steps, batch_size)

            local_policy_values = torch.zeros(opt.num_steps, batch_size)
            local_policy_log_probs = torch.zeros(opt.num_steps, batch_size)
            local_policy_rewards = torch.zeros(opt.num_steps, batch_size)

            Current_IoUs = torch.zeros(opt.num_steps, batch_size)
            IoUs_outputs = torch.zeros(opt.num_steps, batch_size)
            mask = torch.zeros(opt.num_steps, batch_size)

            #network forward
            for step in range(opt.num_steps):

                if step == 0:
                    hidden_state = torch.zeros(batch_size, 1024).cuda()
                    current_feature = initial_feature
                    current_offset = initial_offset
                    current_offset_norm = initial_offset_norm

                hidden_state, global_policy, global_value, scale_policy, scale_value, left_move_policy, left_move_value, right_move_policy, right_move_value, \
                left_offset_policy, left_offset_value, right_offset_policy, right_offset_value, iou_out = net(global_feature, current_feature, sentence, current_offset_norm, hidden_state)

                global_policy_prob = F.softmax(global_policy, dim=1)
                #
                # if batch_idx %100 == 0:
                #     print(global_policy_prob)

                if global_flag == True: # train the global_layer
                    global_policy_log_prob = F.log_softmax(global_policy, dim=1)
                    global_policy_entropy = -(global_policy_log_prob * global_policy_prob).sum(1)
                    global_policy_entropies[step,:] = global_policy_entropy

                    global_policy_action = global_policy_prob.multinomial(num_samples=1).data
                    global_policy_log_prob = global_policy_log_prob.gather(1, global_policy_action)
                    global_policy_action = global_policy_action.cpu().numpy()[:, 0]

                else: # free the global_layer, train the local layer
                    global_policy_action = global_policy_prob.max(1, keepdim=True)[1].data.cpu().numpy()[:, 0]

                # assign global policy to_sub_policy
                current_offset_start = np.zeros(batch_size, dtype=np.int16)
                current_offset_end = np.zeros(batch_size, dtype=np.int16)
                abnormal_done = torch.ones(batch_size)
                update_offset = torch.zeros(batch_size, 2)
                update_offset_norm = torch.zeros(batch_size, 2)

                local_policy_log_prob = torch.zeros(batch_size)
                local_policy_entropy = torch.zeros(batch_size)
                local_policy_value = torch.zeros(batch_size)

                if global_flag == True:
                    golbal_all_norm = torch.zeros(batch_size, 5, 2).cuda()

                for i in  range(batch_size):

                    if global_flag == True:  # train the global_layer
                        scale_policy_prob = F.softmax(scale_policy[i], dim=0)
                        scale_policy_action = scale_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        _, _, _, golbal_all_norm[i][0], _ = determine_scale_range(scale_policy_action, current_offset[i], num_units[i])
                        # also compute other policy results to get the reward
                        left_move_policy_prob = F.softmax(left_move_policy[i], dim=0)
                        left_move_policy_action = left_move_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        _, _, _, golbal_all_norm[i][1], _ = determine_left_move_range(left_move_policy_action, current_offset[i], ten_unit[i], num_units[i])
                        # also compute other policy results to get the reward
                        right_move_policy_prob = F.softmax(right_move_policy[i], dim=0)
                        right_move_policy_action = right_move_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        _, _, _, golbal_all_norm[i][2], _ = determine_right_move_range(right_move_policy_action, current_offset[i], ten_unit[i], num_units[i])
                        left_offset_policy_prob = F.softmax(left_offset_policy[i], dim=0)
                        left_offset_policy_action = left_offset_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        _, _, _, golbal_all_norm[i][3], _ = determine_left_offset_range(left_offset_policy_action, current_offset[i], num_units[i])

                        right_offset_policy_prob = F.softmax(right_offset_policy[i], dim=0)
                        right_offset_policy_action = right_offset_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        _, _, _, golbal_all_norm[i][4], _ = determine_right_offset_range(right_offset_policy_action, current_offset[i], num_units[i])

                    if global_policy_action[i] == 0:

                        scale_policy_prob = F.softmax(scale_policy[i], dim=0)

                        if global_flag == True:  # train the global_layer
                            scale_policy_action = scale_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        else:
                            scale_policy_log_prob = F.log_softmax(scale_policy[i], dim=0)
                            local_policy_entropy[i] = -(scale_policy_log_prob * scale_policy_prob).sum()

                            scale_policy_action = scale_policy_prob.multinomial(num_samples=1).data
                            local_policy_log_prob[i] = scale_policy_log_prob.gather(0, scale_policy_action)
                            scale_policy_action = scale_policy_action.cpu().numpy()[0]
                            local_policy_value[i] = scale_value[i]

                        current_offset_start[i], current_offset_end[i], update_offset[i], update_offset_norm[i], abnormal_done[i] = determine_scale_range(
                            scale_policy_action, current_offset[i], num_units[i])

                    elif global_policy_action[i] ==1:
                        left_move_policy_prob = F.softmax(left_move_policy[i], dim=0)

                        if global_flag == True:  # train the global_layer
                            left_move_policy_action = left_move_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        else:
                            left_move_policy_log_prob = F.log_softmax(left_move_policy[i], dim=0)
                            local_policy_entropy[i] = -(left_move_policy_log_prob * left_move_policy_prob).sum()
                            left_move_policy_action = left_move_policy_prob.multinomial(num_samples=1).data
                            local_policy_log_prob[i] = left_move_policy_log_prob.gather(0, left_move_policy_action)
                            left_move_policy_action = left_move_policy_action.cpu().numpy()[0]
                            local_policy_value[i] = left_move_value[i]

                        current_offset_start[i], current_offset_end[i], update_offset[i], update_offset_norm[i], abnormal_done[i] = determine_left_move_range(
                            left_move_policy_action, current_offset[i], ten_unit[i], num_units[i])

                    elif global_policy_action[i] ==2:
                        right_move_policy_prob = F.softmax(right_move_policy[i], dim=0)

                        if global_flag == True:  # train the global_layer
                            right_move_policy_action = right_move_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        else:
                            right_move_policy_log_prob = F.log_softmax(right_move_policy[i], dim=0)
                            local_policy_entropy[i] = -(right_move_policy_log_prob * right_move_policy_prob).sum()
                            right_move_policy_action = right_move_policy_prob.multinomial(num_samples=1).data
                            local_policy_log_prob[i] = right_move_policy_log_prob.gather(0, right_move_policy_action)
                            right_move_policy_action = right_move_policy_action.cpu().numpy()[0]
                            local_policy_value[i] = right_move_value[i]

                        current_offset_start[i], current_offset_end[i], update_offset[i], update_offset_norm[i], abnormal_done[i] = determine_right_move_range(
                            right_move_policy_action, current_offset[i], ten_unit[i], num_units[i])


                    elif global_policy_action[i] == 3:

                        left_offset_policy_prob = F.softmax(left_offset_policy[i], dim=0)
                        if global_flag == True:  # train the global_layer
                            left_offset_policy_action = left_offset_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        else:
                            left_offset_policy_log_prob = F.log_softmax(left_offset_policy[i], dim=0)
                            local_policy_entropy[i] = -(left_offset_policy_log_prob * left_offset_policy_prob).sum()

                            left_offset_policy_action = left_offset_policy_prob.multinomial(num_samples=1).data
                            local_policy_log_prob[i] = left_offset_policy_log_prob.gather(0, left_offset_policy_action)
                            left_offset_policy_action = left_offset_policy_action.cpu().numpy()[0]
                            local_policy_value[i] = left_offset_value[i]

                        current_offset_start[i], current_offset_end[i], update_offset[i], update_offset_norm[i], abnormal_done[i] = determine_left_offset_range(
                            left_offset_policy_action, current_offset[i], num_units[i])
                    else:

                        right_offset_policy_prob = F.softmax(right_offset_policy[i], dim=0)
                        if global_flag == True:  # train the global_layer
                            right_offset_policy_action = right_offset_policy_prob.max(0, keepdim=True)[1].data.cpu().numpy()
                        else:
                            right_offset_policy_log_prob = F.log_softmax(right_offset_policy[i], dim=0)
                            local_policy_entropy[i] = -(right_offset_policy_log_prob * right_offset_policy_prob).sum()

                            right_offset_policy_action = right_offset_policy_prob.multinomial(num_samples=1).data
                            local_policy_log_prob[i] = right_offset_policy_log_prob.gather(0, right_offset_policy_action)
                            right_offset_policy_action = right_offset_policy_action.cpu().numpy()[0]
                            local_policy_value[i] = right_offset_value[i]

                        current_offset_start[i], current_offset_end[i], update_offset[i], update_offset_norm[i], abnormal_done[i] = determine_right_offset_range(
                            right_offset_policy_action, current_offset[i], num_units[i])


                current_offset = update_offset.cuda()
                current_offset_norm = update_offset_norm.cuda()

                if step == 0:
                    Previou_IoU = calculate_RL_IoU_batch(initial_offset_norm, offset_norm)
                else:
                    Previou_IoU = current_IoU

                mask[step, :] = abnormal_done

                current_IoU = calculate_RL_IoU_batch(current_offset_norm, offset_norm)
                Current_IoUs[step, :] = Previou_IoU #Previou_IoU  #current_IoU
                IoUs_outputs[step, :] = iou_out.squeeze(1)

                if global_flag == True:
                    global_all_IoU = calculate_global_RL_IoU_batch(golbal_all_norm, offset_norm)

                # print("step: %d, action:%d, Previou_IoU: %f, current_IoU: %f"  %(step+1, action, Previou_IoU, current_IoU))

                current_feature = torch.zeros_like(initial_feature).cuda()

                for i in range(batch_size):
                    abnormal = abnormal_done[i]
                    if abnormal == 1:
                        current_feature_med = original_feats[i][(current_offset_start[i]):(current_offset_end[i]+1)]
                        feature_length = len(current_feature_med)
                        idx = choose_ten_frame(feature_length)
                        initial_feature_1 = current_feature_med[idx[0]]
                        initial_feature_2 = current_feature_med[idx[1]]
                        initial_feature_3 = current_feature_med[idx[2]]
                        initial_feature_4 = current_feature_med[idx[3]]
                        initial_feature_5 = current_feature_med[idx[4]]
                        initial_feature_6 = current_feature_med[idx[5]]
                        initial_feature_7 = current_feature_med[idx[6]]
                        initial_feature_8 = current_feature_med[idx[7]]
                        initial_feature_9 = current_feature_med[idx[8]]
                        initial_feature_10 = current_feature_med[idx[9]]
                        initial_feature_concate = torch.cat((initial_feature_1, initial_feature_2, initial_feature_3, initial_feature_4, initial_feature_5, \
                                                             initial_feature_6, initial_feature_7, initial_feature_8, initial_feature_9, initial_feature_10), 0)
                        current_feature[i] = initial_feature_concate

                if global_flag == True:
                    global_reward = calculate_root_reward_batch(Previou_IoU, current_IoU, global_all_IoU)
                else:
                    reward = calculate_leaf_reward_batch(Previou_IoU, current_IoU, step+1)

                if global_flag == True:
                    global_policy_values[step, :] = global_value.squeeze(1)
                    global_policy_log_probs[step, :] = global_policy_log_prob.squeeze(1)
                    global_policy_rewards[step, :] = global_reward  # should determine
                else:
                    local_policy_entropies[step, :] = local_policy_entropy
                    local_policy_values[step, :] = local_policy_value
                    local_policy_log_probs[step, :] = local_policy_log_prob
                    local_policy_rewards[step, :] = reward


            mask_iou_pos = Current_IoUs > 0

            label_iou_for_loss_positive = Current_IoUs[mask_iou_pos]
            output_iou_for_loss_positive = IoUs_outputs[mask_iou_pos]
            iou_loss = regression_loss_func(output_iou_for_loss_positive, label_iou_for_loss_positive)

            if global_flag == True:
                global_policy_loss = 0
                global_value_loss = 0

                idx = 0
                for j in range(
                        batch_size):
                    mask_one = mask[:, j]
                    index = opt.num_steps
                    for i in range(opt.num_steps):
                        if mask_one[i] == 0:
                            index = i + 1
                            break

                    for k in reversed(range(index)):
                        if k == index - 1:
                            global_R = opt.gamma * global_policy_values[k][j] + global_policy_rewards[k][j]
                        else:
                            global_R = opt.gamma * global_R + global_policy_rewards[k][j]

                        global_advantage = global_R - global_policy_values[k][j]
                        global_value_loss = global_value_loss + global_advantage.pow(2)
                        global_policy_loss = global_policy_loss - global_policy_log_probs[k][j] * global_advantage - opt.entropy_coef * global_policy_entropies[k][j]
                        idx += 1

                global_policy_loss /= idx
                global_value_loss /= idx

                global_policy_loss_epoch.append(global_policy_loss.item())
                global_value_loss_epoch.append(global_value_loss.item())
                global_total_rewards_epoch.append(global_policy_rewards.sum())

            else:
                policy_loss = 0
                value_loss = 0

                idx = 0
                for j in range(
                        batch_size):
                    mask_one = mask[:, j]
                    index = opt.num_steps
                    for i in range(opt.num_steps):
                        if mask_one[i] == 0:
                            index = i + 1
                            break

                    for k in reversed(range(index)):
                        if k == index - 1:
                            R = opt.gamma * local_policy_values[k][j] + local_policy_rewards[k][j]
                        else:
                            R = opt.gamma * R + local_policy_rewards[k][j]

                        advantage = R - local_policy_values[k][j]
                        value_loss = value_loss + advantage.pow(2)
                        policy_loss = policy_loss - local_policy_log_probs[k][j] * advantage - opt.entropy_coef * \
                                      local_policy_entropies[k][j]
                        idx += 1

                policy_loss /= idx
                value_loss /= idx
                local_policy_loss_epoch.append(policy_loss.item())
                local_value_loss_epoch.append(value_loss.item())
                total_rewards_epoch.append(local_policy_rewards.sum())

            iou_loss_epoch.append(iou_loss.item())

            optimizer.zero_grad()
            # (policy_loss + value_loss + iou_loss).backward(retain_graph = True)
            if global_flag == True:
                (global_policy_loss + global_value_loss +iou_loss).backward(retain_graph = True)
            else:
                (policy_loss + value_loss +iou_loss).backward(retain_graph = True)
            optimizer.step()

            if global_flag == True:
                print("Train Epoch: %d | Index: %d | global_policy_loss: %f" % (epoch, batch_idx + 1, global_policy_loss.item()))
                print("Train Epoch: %d | Index: %d | global_value_loss: %f" % (epoch, batch_idx + 1, global_value_loss.item()))
            else:
                print("Train Epoch: %d | Index: %d | local_policy loss: %f" % (epoch, batch_idx + 1, policy_loss.item()))
                print("Train Epoch: %d | Index: %d | local_value_loss: %f" % (epoch, batch_idx + 1, value_loss.item()))

            print("Train Epoch: %d | Index: %d | iou_loss: %f" % (epoch, batch_idx+1, iou_loss.item()))

            iteration +=1

            if iteration >0 and iteration % opt.switch_iteration == 0:

                if global_flag == True:

                    ave_global_policy_loss = sum(global_policy_loss_epoch) / len(global_policy_loss_epoch)
                    ave_global_policy_loss_all.append(ave_global_policy_loss)
                    print("Average global Policy Loss for Train iteration %d : %f" % (iteration, ave_global_policy_loss))
                    global_policy_loss_epoch =[]

                    ave_global_value_loss = sum(global_value_loss_epoch) / len(global_value_loss_epoch)
                    ave_global_value_loss_all.append(ave_global_value_loss)
                    print("Average global Value Loss for Train iteration %d : %f" % (iteration, ave_global_value_loss))
                    global_value_loss_epoch = []

                    ave_global_total_rewards = sum(global_total_rewards_epoch) / len(global_total_rewards_epoch)
                    ave_global_total_rewards_all.append(ave_global_total_rewards)
                    print("Average Global Total reward for Train iteration %d: %f" % (iteration, ave_global_total_rewards))
                    global_total_rewards_epoch = []

                    with open(path + "/iteration_ave_global_total_rewards.pkl", "wb") as file:
                        pickle.dump(ave_global_total_rewards_all, file)
                    # plot the val loss vs epoch and save to disk:
                    x = np.arange(1, len(ave_global_total_rewards_all) + 1)
                    plt.figure(1)
                    plt.plot(x, ave_global_total_rewards_all, "r-")
                    plt.ylabel("Rewards")
                    plt.xlabel("Iteration")
                    plt.title("Average Global Reward iteration")
                    plt.xticks(fontsize=8)
                    plt.savefig(path + "/iteration_ave_global_total_rewards.png")
                    plt.close(1)

                    with open(path + "/iteration_ave_global_policy_loss.pkl", "wb") as file:
                        pickle.dump(ave_global_policy_loss_all, file)
                    # plot the val loss vs epoch and save to disk:
                    plt.figure(1)
                    plt.plot(x, ave_global_policy_loss_all, "r-")
                    plt.ylabel("Loss")
                    plt.xlabel("Iteration")
                    plt.title("Average global Policy Loss iteration")
                    plt.xticks(fontsize=8)
                    plt.savefig(path+ "/iteration_ave_global_policy_loss.png")
                    plt.close(1)

                    with open(path + "/iteration_ave_global_value_loss.pkl", "wb") as file:
                        pickle.dump(ave_global_value_loss_all, file)
                    # plot the val loss vs epoch and save to disk:
                    plt.figure(1)
                    plt.plot(x, ave_global_value_loss_all, "r-")
                    plt.ylabel("Loss")
                    plt.xlabel("Iteration")
                    plt.title("Average global Value Loss iteration")
                    plt.xticks(fontsize=8)
                    plt.savefig(path + "/iteration_ave_global_value_loss.png")
                    plt.close(1)


                else:
                    ave_policy_loss = sum(local_policy_loss_epoch) / len(local_policy_loss_epoch)
                    ave_policy_loss_all.append(ave_policy_loss)
                    print("Average Policy Loss for Train iteration %d : %f" % (iteration, ave_policy_loss))
                    local_policy_loss_epoch = []

                    ave_value_loss = sum(local_value_loss_epoch) / len(local_value_loss_epoch)
                    ave_value_loss_all.append(ave_value_loss)
                    print("Average Value Loss for Train iteration %d : %f" % (iteration, ave_value_loss))
                    local_value_loss_epoch = []

                    ave_total_rewards_epoch = sum(total_rewards_epoch) / len(total_rewards_epoch)
                    ave_total_rewards_all.append(ave_total_rewards_epoch)
                    print("Average Total reward for Train iteration %d: %f" % (iteration, ave_total_rewards_epoch))
                    total_rewards_epoch = []

                    with open(path + "/iteration_ave_reward.pkl", "wb") as file:
                        pickle.dump(ave_total_rewards_all, file)
                    # plot the val loss vs epoch and save to disk:
                    x = np.arange(1, len(ave_total_rewards_all) + 1)
                    plt.figure(1)
                    plt.plot(x, ave_total_rewards_all, "r-")
                    plt.ylabel("Rewards")
                    plt.xlabel("Iteration")
                    plt.title("Average Reward iteration")
                    plt.xticks(fontsize=8)
                    plt.savefig(path + "/iteration_ave_reward.png")
                    plt.close(1)

                    with open(path + "/iteration_ave_policy_loss.pkl", "wb") as file:
                        pickle.dump(ave_policy_loss_all, file)
                    # plot the val loss vs epoch and save to disk:
                    plt.figure(1)
                    plt.plot(x, ave_policy_loss_all, "r-")
                    plt.ylabel("Loss")
                    plt.xlabel("Iteration")
                    plt.title("Average Policy Loss iteration")
                    plt.xticks(fontsize=8)
                    plt.savefig(path+ "/iteration_ave_policy_loss.png")
                    plt.close(1)

                    with open(path + "/iteration_ave_value_loss.pkl", "wb") as file:
                        pickle.dump(ave_value_loss_all, file)
                    # plot the val loss vs epoch and save to disk:
                    plt.figure(1)
                    plt.plot(x, ave_value_loss_all, "r-")
                    plt.ylabel("Loss")
                    plt.xlabel("Iteration")
                    plt.title("Average Value Loss iteration")
                    plt.xticks(fontsize=8)
                    plt.savefig(path + "/iteration_ave_value_loss.png")
                    plt.close(1)

                ave_iou_loss = sum(iou_loss_epoch) / len(iou_loss_epoch)
                ave_iou_loss_all.append(ave_iou_loss)
                print("Average Iou Loss for Train iteration %d : %f" % (iteration, ave_iou_loss))
                iou_loss_epoch = []

                with open(path + "/iteration_ave_iou_loss.pkl", "wb") as file:
                    pickle.dump(ave_iou_loss_all, file)
                x = np.arange(1, len(ave_iou_loss_all) + 1)
                # plot the val loss vs epoch and save to disk:
                plt.figure(1)
                plt.plot(x, ave_iou_loss_all, "r-")
                plt.ylabel("Loss")
                plt.xlabel("Iteration")
                plt.title("Average Iou Loss iteration")
                plt.xticks(fontsize=8)
                plt.savefig(path + "/iteration_ave_iou_loss.png")
                plt.close(1)

        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }

        savepath = os.path.join(path, "ckpt")
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        torch.save(state, os.path.join(savepath,'epoch_'+ str(epoch) +'_model.t7'))
        print("save epoch" )
        print("Time", time.time() - start)


if __name__ == '__main__':
    start_epoch = 0
    total_epoch = 500
    ave_global_policy_loss_all = []
    ave_global_value_loss_all = []
    ave_policy_loss_all = []
    ave_value_loss_all = []
    ave_total_rewards_all =[]
    ave_global_total_rewards_all = []
    ave_iou_loss_all = []
    train(start_epoch, total_epoch)


