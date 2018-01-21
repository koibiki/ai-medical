import random
import pandas as pd
import numpy as np
import math


def create_sample(data):
    randoms = []
    for index in range(len(data)):
        random_value = [item * (0.97 + 0.06 * random.random()) for item in data.iloc[index].values]
        random_series = pd.Series(random_value, index=data.columns)
        randoms.append(random_series)
    return pd.DataFrame(randoms).reset_index()


def create_scale_feature(data):
    new_data = data
    columns = data.columns
    for index in range(len(columns)):
        if index == 0:
            continue
        for j in range(index + 1, len(columns)):
            new_data.insert(new_data.shape[1], columns[j] + '_' + columns[index], data.iloc[:, j]/data.iloc[:, index])
    return new_data


def fix_min(data):
    for index in range(len(data)):
        column_values = data.iloc[:, index]
        min_value = 0
        if column_values.min() == 0:
            sort = column_values.sort_values()
            for item in sort:
                if item > 0:
                    min_value = item
                    break
        for i in column_values.index:
            if data.ix[i, index] == 0:
                data.ix[i, index] = min_value
    return data


def fix_min_all(data):
    for index in range(data.shape[1]):
        print(index)
        column_values = data.ix[:, index]
        min_value = 0
        if column_values.min() == 0:
            sort = column_values.sort_values()
            for item in sort:
                if item > 0:
                    min_value = item
                    break
        for i in column_values.index:
            if data.ix[i, index] == 0:
                data.ix[i, index] = min_value
    return data


def normalize_feature(data, f_min=0, f_max=100):
    d_min, d_max = min(data), max(data)
    factor = (f_max - f_min)/(d_max - d_min)
    normalized = f_min + (data - d_min) * factor
    return normalized, factor


def normalize_data_frame(df, start_index):
    factors = {}
    for index in range(start_index, df.shape[1]):
        df.iloc[:, index], factor = normalize_feature(data=df.iloc[:, index])
        factors[index] = factor
    return df, factors


def get_euclidean_metric(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def get_cosine(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    return np_vec1.dot(np_vec2)/(math.sqrt((np_vec1 ** 2).sum()) * math.sqrt((np_vec2 ** 2).sum()))


def get_cosine_angle(vec1, vec2):
    return math.acos(get_cosine(vec1, vec2)) / math.pi * 180
