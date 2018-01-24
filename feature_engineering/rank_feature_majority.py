import pandas as pd
import numpy as np

from utils import search_min_level_all, search_max_level_all


def rank_feature_majority_all(train, test):
    print("执行 rank_feature_majority")
    train_test = pd.concat([train, test], axis=0)
    min_dict = search_min_level_all(train_test, 3)
    max_dict = search_max_level_all(train_test, 3)
    columns = train_test.columns
    for index in range(3, len(columns)):
        column = columns[index]
        train_test['rank_' + column] = \
            train_test[columns[index]].apply(lambda x: rank_feature_rule(x, min_dict[column], max_dict[column]))
    train_num_rank = train_test.iloc[0:train.shape[0], :]
    test_num_rank = train_test.iloc[train.shape[0]:, :]
    return train_num_rank, test_num_rank


def rank_feature_majority_train_valid_test(X_train, X_valid, test):
    train_valid_test = pd.concat([X_train, X_valid, test], axis=0)
    min_dict = search_min_level_all(train_valid_test, 3)
    max_dict = search_max_level_all(train_valid_test, 3)
    columns = train_valid_test.columns
    for index in range(3, len(columns)):
        column = columns[index]
        train_valid_test['rank_' + column] = \
            train_valid_test[columns[index]].apply(lambda x: rank_feature_rule(x, min_dict[column], max_dict[column]))
    train_num_rank = train_valid_test.iloc[0:X_train.shape[0], :]
    valid_num_rank = train_valid_test.iloc[X_train.shape[0]:(X_train.shape[0] + X_valid.shape[0]), :]
    test_num_rank = train_valid_test.iloc[(X_train.shape[0] + X_valid.shape[0]):, :]
    return train_num_rank, valid_num_rank, test_num_rank


def rank_feature_rule(data, min_level, max_level):
    if np.isnan(data):
        return data
    elif data < min_level:
        return 0
    elif (data >= min_level) & (data < min_level + (max_level - min_level) / 9):
        return 1
    elif (data >= min_level + (max_level - min_level) / 9) & (data < min_level + (max_level - min_level) * 2 / 9):
        return 2
    elif (data >= min_level + (max_level - min_level) * 2 / 9) & (data < min_level + (max_level - min_level) * 3 / 9):
        return 3
    elif (data >= min_level + (max_level - min_level) * 3 / 9) & (data < min_level + (max_level - min_level) * 4 / 9):
        return 4
    elif (data >= min_level + (max_level - min_level) * 4 / 9) & (data < min_level + (max_level - min_level) * 5 / 9):
        return 5
    elif (data >= min_level + (max_level - min_level) * 5 / 9) & (data < min_level + (max_level - min_level) * 6 / 9):
        return 6
    elif (data >= min_level + (max_level - min_level) * 6 / 9) & (data < min_level + (max_level - min_level) * 7 / 9):
        return 7
    elif (data >= min_level + (max_level - min_level) * 7 / 9) & (data < min_level + (max_level - min_level) * 8 / 9):
        return 8
    elif (data >= min_level + (max_level - min_level) * 8 / 9) & (data < max_level):
        return 9
    elif data >= max_level:
        return 10

def rank_feature_count(train, test):
    train_test = pd.concat([train, test], axis=0)
    for i in range(1, 11, 1):
        train_test['n' + str(i)] = (train_test == i).sum(axis=1)
    train_rank_count = train_test.iloc[0:train.shape[0], :]
    test_rank_count = train_test.iloc[train.shape[0]:, :]
    return train_rank_count, test_rank_count
