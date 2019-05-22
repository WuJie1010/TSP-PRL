" Dataloader of charades-STA dataset for RL algorithm"

import torch
import torch.utils.data
import os
import pickle
import numpy as np
import math
from utils import *
import random
import glob

class Charades_Train_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.unit_size = 16
        self.feats_dimen = 4096
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 4096
        self.sent_vec_dim = 4800
        self.sliding_clip_path = "./Dataset/Charades/Charades_localfeature_npy/"#"./Dataset/Charades/all_fc6_unit16_overlap0.5/"
        self.clip_sentence_pairs_iou_all = pickle.load(open("./Dataset/Charades/ref_info/charades_rl_train_feature.pkl"))

        self.num_samples_iou = len(self.clip_sentence_pairs_iou_all)
        print(self.num_samples_iou, "iou clip-sentence pairs are readed")  # 49442

        self.movie_name_id = []

        self.global_feature_all = []
        self.original_feats_all = []
        self.initial_feature_all = []
        self.ten_unit_all = []
        self.initial_offset_start_all = []
        self.initial_offset_end_all = []
        self.initial_offset_start_norm_all = []
        self.initial_offset_end_norm_all = []
        self.num_units_all = []

        movie_name_list = []
        for i in range(self.num_samples_iou):
            if (i+1) % 1000 == 0:
                print("%d/%d read" %((i+1), self.num_samples_iou))
            movie_name = self.clip_sentence_pairs_iou_all[i]['video']
            end = self.clip_sentence_pairs_iou_all[i]['frames_num'] +1

            if movie_name not in movie_name_list:
                global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, \
                initial_offset_end_norm, num_units = self.read_video_level_feats(movie_name, end)

                self.global_feature_all.append(global_feature)
                self.original_feats_all.append(original_feats)
                self.initial_feature_all.append(initial_feature)
                self.ten_unit_all.append(ten_unit)
                self.initial_offset_start_all.append(initial_offset_start)
                self.initial_offset_end_all.append(initial_offset_end)
                self.initial_offset_start_norm_all.append(initial_offset_start_norm)
                self.initial_offset_end_norm_all.append(initial_offset_end_norm)
                self.num_units_all.append(num_units)
                movie_name_list.append(movie_name)
            self.movie_name_id.append(len(movie_name_list)-1)
            # print(self.movie_name_id)

    def read_video_level_feats(self, movie_name, end):
        # read unit level feats by just passing the start and end number
        unit_size = 16
        feats_dimen = 4096 *3
        start = 1
        num_units = (end - start) / unit_size
        # print(start, end, num_units)
        curr_start = 1

        ten_unit = num_units / 10
        four_unit = num_units / 4
        oneinfour_unit = four_unit
        threeinfour_unit = num_units - four_unit

        start_end_list = []
        while (curr_start + unit_size <= end):
            start_end_list.append((curr_start, curr_start + unit_size))
            curr_start += unit_size

        original_feats = np.zeros([300, feats_dimen], dtype=np.float32)
        original_feats_1 = np.zeros([num_units, feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            one_feat = np.load(self.sliding_clip_path + movie_name + "_" + str(curr_s) + ".0_" + str(curr_e) + ".0.npy")
            original_feats[k] = one_feat
            original_feats_1[k] = one_feat
        # print(np.shape(original_feats))
        global_feature = np.mean(original_feats_1, axis=0)

        initial_feature = original_feats[(oneinfour_unit-1):(threeinfour_unit)]

        feature_length = len(initial_feature)
        idx = choose_four_frame(feature_length)
        initial_feature_1 = initial_feature[idx[0]]
        initial_feature_2 = initial_feature[idx[1]]
        initial_feature_3 = initial_feature[idx[2]]
        initial_feature_4 = initial_feature[idx[3]]
        initial_feature_concate = np.concatenate(
            [initial_feature_1, initial_feature_2, initial_feature_3, initial_feature_4], axis=0)

        initial_feature = initial_feature_concate

        initial_offset_start = oneinfour_unit-1
        initial_offset_end = threeinfour_unit - 1

        initial_offset_start_norm = initial_offset_start / float(num_units-1)
        initial_offset_end_norm = initial_offset_end / float(num_units-1)

        return global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units

    def __getitem__(self, index):
        # print(index)
        offset = np.zeros(2, dtype=np.float32)
        offset_norm = np.zeros(2, dtype=np.float32)
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)

        samples = self.clip_sentence_pairs_iou_all[index]

        proposal_or_sliding_window = samples['proposal_or_sliding_window']

        movie_name_id = self.movie_name_id[index]
        global_feature = self.global_feature_all[movie_name_id]
        original_feats = self.original_feats_all[movie_name_id]
        initial_feature = self.initial_feature_all[movie_name_id]
        ten_unit = self.ten_unit_all[movie_name_id]
        initial_offset_start = self.initial_offset_start_all[movie_name_id]
        initial_offset_end = self.initial_offset_end_all[movie_name_id]
        initial_offset_start_norm = self.initial_offset_start_norm_all[movie_name_id]
        initial_offset_end_norm = self.initial_offset_end_norm_all[movie_name_id]
        num_units = self.num_units_all[movie_name_id]

        sentence = samples['sent_skip_thought_vec'][0][0]
        # print(np.shape(sentence))
        offset_start = samples['offset_start']
        offset_end = samples['offset_end']
        offset_start_norm = samples['offset_start_norm']
        offset_end_norm = samples['offset_end_norm']

        # offest
        offset[0] = offset_start
        offset[1] = offset_end

        offset_norm[0] = offset_start_norm
        offset_norm[1] = offset_end_norm

        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        return global_feature, original_feats, initial_feature, sentence, offset_norm, initial_offset, initial_offset_norm, ten_unit, num_units

    def __len__(self):
        return self.num_samples_iou

class Charades_Test_dataset(torch.utils.data.Dataset):
    def __init__(self):

        # il_path: image_label_file path
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 4096
        self.feats_dimen = 4096
        self.unit_size = 16
        self.semantic_size = 4800
        self.sliding_clip_path =  "./Dataset/Charades/Charades_localfeature_npy/"#"./Dataset/Charades/all_fc6_unit16_overlap0.5/"
        self.index_in_epoch = 0
        self.spacy_vec_dim = 300
        self.sent_vec_dim = 4800
        self.epochs_completed = 0

        self.clip_sentence_pairs = pickle.load(open("./Dataset/Charades/ref_info/charades_sta_test_semantic_sentence_VP_sub_obj.pkl"))
        self.num_samples_iou = len(self.clip_sentence_pairs)
        print str(len(self.clip_sentence_pairs)) + " test videos are readed"  # 1334

        movie_names_set = set()
        for ii in self.clip_sentence_pairs:
            for iii in self.clip_sentence_pairs[ii]:
                clip_name = iii
                movie_name = ii
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
        self.movie_names = list(movie_names_set)

    def read_video_level_feats(self, movie_name, end):
        # read unit level feats by just passing the start and end number
        unit_size = 16
        feats_dimen = 4096 *3
        start = 1
        num_units = (end - start) / unit_size
        # print(start, end, num_units)
        curr_start = 1

        ten_unit = num_units / 10
        four_unit = num_units / 4
        oneinfour_unit = four_unit
        threeinfour_unit = num_units - four_unit

        start_end_list = []
        while (curr_start + unit_size <= end):
            start_end_list.append((curr_start, curr_start + unit_size))
            curr_start += unit_size

        # original_feats = np.zeros([num_units, feats_dimen], dtype=np.float32)
        original_feats = np.zeros([num_units, feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            one_feat = np.load(self.sliding_clip_path + movie_name + "_" + str(curr_s) + ".0_" + str(curr_e) + ".0.npy")
            original_feats[k] = one_feat
        # print(np.shape(original_feats))
        global_feature = np.mean(original_feats, axis=0)

        initial_feature = original_feats[(oneinfour_unit-1):(threeinfour_unit)]

        feature_length = len(initial_feature)

        idx = choose_four_frame(feature_length)
        initial_feature_1 = initial_feature[idx[0]]
        initial_feature_2 = initial_feature[idx[1]]
        initial_feature_3 = initial_feature[idx[2]]
        initial_feature_4 = initial_feature[idx[3]]
        initial_feature_concate = np.concatenate([initial_feature_1, initial_feature_2, initial_feature_3, initial_feature_4], axis=0)
        initial_feature = initial_feature_concate


        initial_offset_start = oneinfour_unit-1
        initial_offset_end = threeinfour_unit - 1

        initial_offset_start_norm = initial_offset_start / float(num_units-1)
        initial_offset_end_norm = initial_offset_end / float(num_units-1)

        return global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units

    def __getitem__(self, index):
        movie_name = self.movie_names[index]

        # load unit level feats and sentence vector
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)
        movie_clip_sentences = []

        checkpoint_paths = glob.glob(self.sliding_clip_path + movie_name + "_*")
        checkpoint_file_name_ints = [int(float(x.split('/')[-1].split('.npy')[0].split('_')[-1]))
                                     for x in checkpoint_paths]
        end = max(checkpoint_file_name_ints)

        global_feature, original_feats, initial_feature, ten_unit, initial_offset_start, initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, num_units\
            = self.read_video_level_feats(movie_name, end)

        ten_unit = np.array(ten_unit)
        num_units = np.array(num_units)

        for dict_2nd in self.clip_sentence_pairs[movie_name]:
            for dict_3rd in self.clip_sentence_pairs[movie_name][dict_2nd]:

                sentence_vec_ = dict_3rd['sent_skip_thought_vec'][0][0, :self.sent_vec_dim]
                movie_clip_sentences.append((dict_2nd, sentence_vec_))


        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        return movie_clip_sentences, global_feature, original_feats, initial_feature, initial_offset, initial_offset_norm, ten_unit, num_units


    def __len__(self):
        return self.num_samples_iou