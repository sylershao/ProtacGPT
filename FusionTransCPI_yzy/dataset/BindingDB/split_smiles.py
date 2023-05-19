'''
!/usr/bin/env python
-*- coding:utf-8 -*-
@ProjectName  :Code
@FileName  :Split_SMILES.py
@Time      :2021/7/1 16:12
@Author    :sylershao
'''
import pickle

import pandas as pd
import numpy as np
import time
import collections
from rdkit import Chem
import tqdm


def split_smiles(smiles):
    smiles_level = 0  # 读取层级
    smiles_level_max = 0
    index_level_stat_dict = {}  # {index:[w:这是字母,level,(-1,0,1，由上上一个字母到本字母到区别，默认为0)]
    for i, w in enumerate(smiles):
        state = 0
        if i == 0: state = 1

        if w == '(':
            smiles_level += 1
            if smiles_level_max < smiles_level: smiles_level_max = smiles_level
            state = 1
        if w == ')':
            smiles_level -= 1
            state = -1

        index_level_stat_dict[i] = [w, smiles_level, state]

    # print(index_level_stat_dict)

    # 输出level
    split_result = []

    split_str = ''

    for i in range(len(smiles)):
        # print('index_level_stat_dict[{}]'.format(i),index_level_stat_dict[i])

        if index_level_stat_dict[i][2] == 0:
            split_result[index_level_stat_dict[i][1]][-1] = split_result[index_level_stat_dict[i][1]][-1] + \
                                                            index_level_stat_dict[i][0]
        elif index_level_stat_dict[i][2] == 1:
            if len(split_result) == index_level_stat_dict[i][1]:
                split_result.append([index_level_stat_dict[i][0]])
            else:
                split_result[index_level_stat_dict[i][1]].append(index_level_stat_dict[i][0])
        elif index_level_stat_dict[i][2] == -1:
            split_result[index_level_stat_dict[i][1]][-1] = split_result[index_level_stat_dict[i][1]][-1] + 'R' + \
                                                            index_level_stat_dict[i][0]
    # print(split_result)

    for i, _ in enumerate(split_result):
        for j, ww in enumerate(split_result[i]):
            if '(' in ww: ww = ww.replace('(', '')
            if ')' in ww: ww = ww.replace(')', '')
            split_result[i][j] = ww

    # print(split_result)

    # 返回用于作图的dict
    split_result_dict = {}
    dict_i = {}
    for i in range(len(split_result)):
        _i = len(split_result) - 1 - i
        # print('_i',_i)

        R_count = 0

        dict_i_temp = {}
        for j, a in enumerate(split_result[_i]):
            # dict_i[j] = a
            dict_i_temp[j] = a
            if 'R' in a:
                count_w = 0
                for w in a:
                    if w == 'R': count_w += 1

                dict_i_temp[j] = {}
                dict_a = {}
                dict_i_temp[j][a] = {}
                for c in range(R_count, R_count + count_w, 1):
                    dict_i_temp[j][a][c - R_count] = dict_i[c]
                R_count += count_w

        dict_i = dict_i_temp
        #
        # print(dict_i)
        # print(dict_i_temp)
    return dict_i_temp, split_result

    # for j,b in enumerate(split_result[i]):


def plot_tree_test():
    a = {'no_surfacing':
             {'L1': {'flippers':
                         {'R0': 'no', 'R1': 'yes'}},
              'L0': {'flippers':
                         {'R0': 'why', 'R1': 'no'}},
              'L2': 'yes'}
         }
    b = {'Cc1cccRcc1': {
        0: {'-c2nc3ccccc3cRn2C/C=C/c2cccRcc2\r\nasdasd': {0: '=O', 1: {'CRCN3CRCOc4ccccc43': {0: '=O', 1: '=O'}}}}}}
    # treePlotter.createPlot(b, 'test3')


def encoder(data_list, data_index):
    res = []
    for j in range(data_index[-1] + 1):
        word_list = []
        # print(data_index[-1])
        for i in range(len(data_index)):
            if data_index[i] == j:
                word_list.append(int(dict_index_word[data_list[i]]))
            # else:
            #     break
        res.append(word_list)
    # res = torch.nn.utils.rnn.pad_sequence(res, batch_first=True, padding_value=50)
    print(res.__len__())
    return res


def main(input_part):
    part = input_part
    data = pd.read_csv(input_part + '.csv')
    print('read listdata...')
    SMILESs = list(data['SMILES'])
    # Label = list(data['Label'])
    # # 建立二维list存放分割结果
    # split_result_to_pkl = [SMILESs, GI50_Data, []]
    # Statistics_result_dict = {}
    # split_result_to_csv = pd.DataFrame()
    # split_result_to_csv['SMILES'] = SMILESs
    # split_result_to_csv['Label'] = Label
    # for i in range(500):
    #     split_result_to_csv['split_{}'.format(i)] = [np.nan]*len(split_result_to_csv)

    # split_result_to_csv.columns = split_result_to_csv.columns

    df_smiles_list = pd.DataFrame(columns=['index', 'sub_smiles'])

    for smiles_index, smiles in enumerate(tqdm.tqdm(SMILESs)):
        # print('smiles', smiles_index, smiles)
        # print('timecost',(time.time()-times))

        if smiles == smiles:  # 知识点 numpy特性  np.nan != np.nan  (判断是否为  np.nan)
            # SMILES canonicalization | SMILES to Canonical SMILES

            mol = Chem.MolFromSmiles(smiles)
            try:
                canonical_smi = Chem.MolToSmiles(mol)
            except:
                continue
            # if not smiles == canonical_smi:  # 展示 canonicalization 效果
            # print('canonical_smi', canonical_smi)
            # 置换符号
            if 'Cl' in canonical_smi: canonical_smi = canonical_smi.replace('Cl', 'L')
            if 'Br' in canonical_smi: canonical_smi = canonical_smi.replace('Br', 'P')

            # 分割smiles
            result_dict, result_list = split_smiles(canonical_smi)
            # print(result_list)

            # 分割的基础上进行bpe算法
            smiles_list = []
            for layer in result_list:
                smiles_list.extend(layer)
            # print(smiles_list)
            df_smiles = pd.DataFrame()
            df_smiles['index'] = [smiles_index] * len(smiles_list)
            df_smiles['sub_smiles'] = smiles_list

            df_smiles_list = pd.concat([df_smiles_list, df_smiles])
            # print(df_smiles_list)

        # if smiles_index == 1000:break

    df_smiles_list.to_csv(input_part + '_split.sub_smiles.csv')
    # df_smiles_list['sub_smiles'].to_csv('test_split.sub_smiles.noindex.csv', index=None, header=None)


if __name__ == "__main__":

    print('preprocessing...')
    # main("train")
    # main("test")
    # main("val")
    main('mol_fragment_zinc250k/predict')


    # val_df = pd.read_csv("val_split.sub_smiles.csv")
    # train_df = pd.read_csv("train_split.sub_smiles.csv")
    # test_df = pd.read_csv("test_split.sub_smiles.csv")
    print('processing...')
    pred_df = pd.read_csv("mol_fragment_zinc250k/predict_split.sub_smiles.csv")
    # print(val_df['sub_smiles'])
    #
    # val_df_list = list(val_df['sub_smiles'])
    # val_df_index = list(val_df['index'])
    #
    # train_df_list = list(train_df['sub_smiles'])
    # train_df_index = list(train_df['index'])
    #
    # test_df_list = list(test_df['sub_smiles'])
    # test_df_index = list(test_df['index'])

    pred_df_list = list(pred_df['sub_smiles'])
    pred_df_index = list(pred_df['index'])

    # sub_word_list = val_df_list + test_df_list + train_df_list
    sub_word_list = pred_df_index
    # print(sub_word_list)

    word2dict = collections.Counter(sub_word_list)

    demo = list(word2dict.keys())
    print(demo)

    dict_index_word = {}
    for index, word in enumerate(tqdm.tqdm(demo)):
        dict_index_word[word] = index

    print(dict_index_word)

    # dict_index_word = dict(demo3, demo2)
    # print(word2dict)
    # print(dict_index_word)

    # train_res = encoder(train_df_list, train_df_index)
    #
    # test_res = encoder(test_df_list, test_df_index)
    # #
    # val_res = encoder(val_df_list, val_df_index)

    pred_res = encoder(pred_df_list, pred_df_index)
    # print(val_res)
    #
    # df_train_encoder = pd.DataFrame()
    # df_train_encoder['smiles_encoder'] = train_res
    # df_train_encoder.to_pickle('train_split_sub_smiles.pickle')
    #
    # df_test_encoder = pd.DataFrame()
    # df_test_encoder['smiles_encoder'] = test_res
    # df_test_encoder.to_pickle('test_split_sub_smiles.pickle')
    #
    # df_val_encoder = pd.DataFrame()
    # df_val_encoder['smiles_encoder'] = val_res
    # df_val_encoder.to_pickle('val_split_sub_smiles.pickle')

    df_pred_encoder = pd.DataFrame()
    df_pred_encoder['smiles_encoder'] = pred_res
    df_pred_encoder.to_pickle('pred_split_sub_smiles.pickle')
