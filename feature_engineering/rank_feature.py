import pandas as pd


# 与 rank by max 择一执行
def rank_feature(train, test):
    print("执行 Rank Feature")
    train_test = pd.concat([train, test], axis=0)
    train_test_rank = train_test.rank(method='max')
    rank_count = train_test_rank.shape[0]
    train_test_rank[train_test_rank <= rank_count / 10] = 1
    train_test_rank[(rank_count / 10 < train_test_rank) & (train_test_rank <= rank_count / 5)] = 2
    train_test_rank[(rank_count / 5 < train_test_rank) & (train_test_rank <= rank_count * 3 / 10)] = 3
    train_test_rank[(rank_count * 3 / 10 < train_test_rank) & (train_test_rank <= rank_count * 2 / 5)] = 4
    train_test_rank[(rank_count * 2 / 5 < train_test_rank) & (train_test_rank <= rank_count * 1 / 2)] = 5
    train_test_rank[(rank_count * 1 / 2 < train_test_rank) & (train_test_rank <= rank_count * 3 / 5)] = 6
    train_test_rank[(rank_count * 3 / 5 < train_test_rank) & (train_test_rank <= rank_count * 7 / 10)] = 7
    train_test_rank[(rank_count * 7 / 10 < train_test_rank) & (train_test_rank <= rank_count * 4 / 5)] = 8
    train_test_rank[(rank_count * 4 / 5 < train_test_rank) & (train_test_rank <= rank_count * 9 / 10)] = 9
    train_test_rank[rank_count * 9 / 10 < train_test_rank] = 10

    train_num_rank = train_test_rank.iloc[0:train.shape[0], :]
    test_num_rank = train_test_rank.iloc[train.shape[0]:, :]
    return train_num_rank, test_num_rank


# 需要在 rank 之前运行
def rank_feature_by_max(train, test):
    print("执行 Rank Feature by max")
    train_test = pd.concat([train, test], axis=0)
    train_test_rank_standard = (train_test - train_test.min())/(train_test.max() - train_test.min())
    train_test_rank_standard[train_test_rank_standard <= 0.1] = 1
    train_test_rank_standard[(0.1 < train_test_rank_standard) & (train_test_rank_standard <= 0.2)] = 2
    train_test_rank_standard[(0.2 < train_test_rank_standard) & (train_test_rank_standard <= 0.3)] = 3
    train_test_rank_standard[(0.3 < train_test_rank_standard) & (train_test_rank_standard <= 0.4)] = 4
    train_test_rank_standard[(0.4 < train_test_rank_standard) & (train_test_rank_standard <= 0.5)] = 5
    train_test_rank_standard[(0.5 < train_test_rank_standard) & (train_test_rank_standard <= 0.6)] = 6
    train_test_rank_standard[(0.6 < train_test_rank_standard) & (train_test_rank_standard <= 0.7)] = 7
    train_test_rank_standard[(0.7 < train_test_rank_standard) & (train_test_rank_standard <= 0.8)] = 8
    train_test_rank_standard[(0.8 < train_test_rank_standard) & (train_test_rank_standard <= 0.9)] = 9
    train_test_rank_standard[(0.9 < train_test_rank_standard) & (train_test_rank_standard <= 1.0)] = 10

    train_num_rank = train_test_rank_standard.iloc[0:train.shape[0], :]
    test_num_rank = train_test_rank_standard.iloc[train.shape[0]:, :]
    return train_num_rank, test_num_rank


def rank_feature_count(train, test):
    train_test = pd.concat([train, test], axis=0)
    for i in range(1, 11, 1):
        train_test['n' + str(i)] = (train_test == i).sum(axis=1)
    train_rank_count = train_test.iloc[0:train.shape[0], :]
    test_rank_count = train_test.iloc[train.shape[0]:, :]
    return train_rank_count, test_rank_count
