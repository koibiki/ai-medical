import numpy as np


# 只能处理数值特征 , 需先分离字符和数值 项
def delete_all_nan(train, test):
    print("执行 Delete nan")
    train_nan_columns = np.where(np.isnan(train))[1]
    test_nan_columns = np.where(np.isnan(test))[1]
    all_nan_columns = mix(train_nan_columns, test_nan_columns)
    train_delete_nan = train.drop(train.columns[all_nan_columns], axis=1)
    test_delete_nan = test.drop(test.columns[all_nan_columns], axis=1)
    return train_delete_nan, test_delete_nan


def delete_nan_columns(data):
    nan_columns = np.where(np.isnan(data))[1]
    nan_columns = get_nan_indexes(nan_columns)
    return nan_columns


def get_nan_indexes(data):
    indexes = []
    for index in data:
        if index not in indexes:
            indexes.append(index)
    return indexes


def mix(train, test):
    train_indexes = get_nan_indexes(train)
    test_indexes = get_nan_indexes(test)
    nan_indexes = list(train_indexes) + list(test_indexes)
    return nan_indexes
