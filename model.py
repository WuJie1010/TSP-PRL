" Model file of Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video "

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)

class HRL(nn.Module):
    def __init__(self):
        " global policy denotes the root policy, left policy denotes the left policy"
        super(HRL, self).__init__()
        self.sentence_embedding_size = 4800
        self.visual_feature_dim = 4096*2
        self.gobal_fc = nn.Linear(self.visual_feature_dim*10, 1024)
        self.local_fc = nn.Linear(self.visual_feature_dim*10, 1024)
        self.sentence_fc = nn.Linear(self.sentence_embedding_size, 1024)
        self.location_fc = nn.Linear(2, 1024)
        self.state_fc = nn.Linear(1024+1024+1024, 1024)

        self.gru = nn.GRUCell(1024, 1024)
        self.global_value = nn.Linear(1024, 1)
        self.global_policy = nn.Linear(1024, 5) #scale, move, offset refine

        self.scale_value = nn.Linear(1024, 1)
        self.scale_policy = nn.Linear(1024, 4) #scale action based on centrel : expand 1.5, expand 2.0; reduce 1.5, reduce 2.0

        self.left_move_value = nn.Linear(1024, 1)
        self.left_move_policy = nn.Linear(1024, 3) #left move action 3 actions based on 10/units

        self.right_move_value = nn.Linear(1024, 1)
        self.right_move_policy = nn.Linear(1024, 3) #right move action 3 actions based on 10/units

        self.left_offset_value = nn.Linear(1024, 1)
        self.left_offset_policy = nn.Linear(1024, 3) #left offset action based on 16 frame, 1 unit, six!

        self.right_offset_value = nn.Linear(1024, 1)
        self.right_offset_policy = nn.Linear(1024, 3) #right offset action based on 16 frame, 1 unit, six!

        self.iou_fc1 = nn.Linear(1024, 256)
        self.iou_fc2 = nn.Linear(256, 128)
        self.iou_fc3 = nn.Linear(128, 1)

        # Initializing weights
        self.apply(weights_init)
        self.global_policy.weight.data = normalized_columns_initializer(self.global_policy.weight.data, 0.01)
        self.global_policy.bias.data.fill_(0)
        self.global_value.weight.data = normalized_columns_initializer(self.global_value.weight.data, 1.0)
        self.global_value.bias.data.fill_(0)

        self.scale_policy.weight.data = normalized_columns_initializer(self.scale_policy.weight.data, 0.01)
        self.scale_policy.bias.data.fill_(0)
        self.scale_value.weight.data = normalized_columns_initializer(self.scale_value.weight.data, 1.0)
        self.scale_value.bias.data.fill_(0)

        self.left_move_policy.weight.data = normalized_columns_initializer(self.left_move_policy.weight.data, 0.01)
        self.left_move_policy.bias.data.fill_(0)
        self.left_move_value.weight.data = normalized_columns_initializer(self.left_move_value.weight.data, 1.0)
        self.left_move_value.bias.data.fill_(0)

        self.right_move_policy.weight.data = normalized_columns_initializer(self.right_move_policy.weight.data, 0.01)
        self.right_move_policy.bias.data.fill_(0)
        self.right_move_value.weight.data = normalized_columns_initializer(self.right_move_value.weight.data, 1.0)
        self.right_move_value.bias.data.fill_(0)

        self.left_offset_policy.weight.data = normalized_columns_initializer(self.left_offset_policy.weight.data, 0.01)
        self.left_offset_policy.bias.data.fill_(0)
        self.left_offset_value.weight.data = normalized_columns_initializer(self.left_offset_value.weight.data, 1.0)
        self.left_offset_value.bias.data.fill_(0)

        self.right_offset_policy.weight.data = normalized_columns_initializer(self.right_offset_policy.weight.data, 0.01)
        self.right_offset_policy.bias.data.fill_(0)
        self.right_offset_value.weight.data = normalized_columns_initializer(self.right_offset_value.weight.data, 1.0)
        self.right_offset_value.bias.data.fill_(0)

    def forward(self, global_feature, local_feature, senetence_feature, location_feature, hidden_state):
        global_feature =self.gobal_fc(global_feature)
        global_feature_norm = F.normalize(global_feature, p=2, dim=1)
        global_feature_norm =  F.relu(global_feature_norm)

        local_feature = self.local_fc(local_feature)
        local_feature_norm = F.normalize(local_feature, p=2, dim=1)
        local_feature_norm = F.relu(local_feature_norm)

        senetence_feature = self.sentence_fc(senetence_feature)
        location_feature = self.location_fc(location_feature)
        location_feature_norm = F.normalize(location_feature, p=2, dim=1)
        location_feature_norm = F.relu(location_feature_norm)

        # local gate-attention
        senetence_feature_norm = F.sigmoid(senetence_feature)
        assert local_feature_norm.size() == senetence_feature_norm.size()
        local_attention_feature = local_feature_norm * senetence_feature_norm

        # global gate-attention
        assert global_feature_norm.size() == senetence_feature_norm.size()
        global_attention_feature = global_feature_norm * senetence_feature_norm

        # location gate-attention
        assert location_feature_norm.size() == senetence_feature_norm.size()
        location_attention_feature = location_feature_norm * senetence_feature_norm

        state_feature = torch.cat([local_attention_feature, global_attention_feature, location_attention_feature], 1)

        state_feature = self.state_fc(state_feature)
        state_feature = F.relu(state_feature)

        hidden_state = self.gru(state_feature, hidden_state)

        global_value = self.global_value(hidden_state)
        global_policy = self.global_policy(hidden_state)

        scale_value = self.scale_value(hidden_state)
        scale_policy = self.scale_policy(hidden_state)

        left_move_value = self.left_move_value(hidden_state)
        left_move_policy = self.left_move_policy(hidden_state)

        right_move_value = self.right_move_value(hidden_state)
        right_move_policy = self.right_move_policy(hidden_state)

        left_offset_value = self.left_offset_value(hidden_state)
        left_offset_policy = self.left_offset_policy(hidden_state)

        right_offset_value = self.right_offset_value(hidden_state)
        right_offset_policy = self.right_offset_policy(hidden_state)

        iou_out = F.relu(self.iou_fc1(state_feature))
        iou_out = F.relu(self.iou_fc2(iou_out))
        iou_out = self.iou_fc3(iou_out)

        return hidden_state, global_policy, global_value, scale_policy, scale_value, left_move_policy, left_move_value, right_move_policy, right_move_value, \
               left_offset_policy, left_offset_value, right_offset_policy, right_offset_value, iou_out

