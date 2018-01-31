import random
import pandas as pd
import numpy as np
import math


def sample_rule(x):
    if np.isnan(x):
        return x
    else:
        random.seed(4)

        return x * (0.97 + 0.06 * random.random())


def create_sample(data):
    randoms = []
    data_sex_age_date = data.iloc[:, :3].reset_index(drop=True)
    data_other = data.iloc[:, 3:].reset_index(drop=True)
    for index in range(len(data_other)):
        random_value = [sample_rule(item) for item in data_other.iloc[index].values]
        random_series = pd.Series(random_value, index=data_other.columns)
        randoms.append(random_series)
    random_data = pd.DataFrame(randoms).reset_index(drop=True)
    return pd.concat([data_sex_age_date, random_data], axis=1)


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
    factor = (f_max - f_min) / (d_max - d_min)
    normalized = f_min + (data - d_min) * factor
    return normalized, factor


def normalize_data_frame(df, start_index=0):
    factors = {}
    for index in range(start_index, df.shape[1]):
        df.iloc[:, index], factor = normalize_feature(data=df.iloc[:, index])
        factors[index] = factor
    return df, factors


def get_euclidean_metric(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def get_cosine(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    return np_vec1.dot(np_vec2) / (math.sqrt((np_vec1 ** 2).sum()) * math.sqrt((np_vec2 ** 2).sum()))


def get_cosine_angle(vec1, vec2):
    return math.acos(get_cosine(vec1, vec2)) / math.pi * 180


def combine_all(arr, start, result, index, group_array):
    for ct in range(start, len(arr) - index + 1):
        result[index - 1] = ct
        if index - 1 == 0:
            copy = result.copy()
            copy.reverse()
            group_array.append(copy)
        else:
            combine_all(arr, ct + 1, result, index - 1, group_array)


# 删除异常点
def delete_error_data(df):
    print(df.shape)
    df = df[(df['f0'] < 200).values | np.isnan(df['f0'])]
    print(df.shape)
    df = df[(df['f1'] < 250).values | np.isnan(df['f1'])]
    print(df.shape)
    df['f2'] = df['f2'].apply(lambda x: x if x < 200 else 200)
    print(df.shape)
    df = df[(df['f3'] < 400).values | np.isnan(df['f3'])]
    df['f3'] = df['f3'].apply(lambda x: x if x < 300 else 300)
    print(df.shape)
    df = df[(df['f4'] < 95).values & (df['f4'] > 60).values | np.isnan(df['f4'])]
    print(df.shape)
    df = df[(df['f6'] > 20).values & (df['f6'] < 50).values | np.isnan(df['f6'])]
    print(df.shape)
    df = df[(df['f7'] < 3).values | np.isnan(df['f7'])]
    print(df.shape)
    df = df[(df['f8'] < 20).values | np.isnan(df['f8'])]
    print(df.shape)
    df = df[(df['f9'] < 12.5).values | np.isnan(df['f9'])]
    print(df.shape)
    df = df[(df['f10'] < 3).values | np.isnan(df['f10'])]
    print(df.shape)
    df = df[(df['f17'] < 2.5).values | np.isnan(df['f17'])]
    print(df.shape)
    # 删除 f19 kf mse 0.97995  predict mse 0.7308
    # 不删除   kf mse 0.95059  predict mse 1.0235
    df = df[(df['f19'] < 12.5).values | np.isnan(df['f19'])]
    print(df.shape)
    df = df[(df['f20'] < 16).values | np.isnan(df['f20'])]
    print(df.shape)
    df = df[(df['f28'] < 600).values | np.isnan(df['f28'])]
    print(df.shape)
    df = df[(df['f32'] > 25).values | np.isnan(df['f32'])]
    print(df.shape)
    df = df[(df['f33'] < 60).values | np.isnan(df['f33'])]
    print(df.shape)
    return df


def filtration(df):
    df['sex'] = df['sex'].apply(lambda x: 1 if np.isnan(x) else x)
    df['f0'] = df['f0'].apply(lambda x: x if (x < 150) | np.isnan(x) else 150)

    df['f1'] = df['f1'].apply(lambda x: x if (x < 200) | np.isnan(x) else 200)

    df['f2'] = df['f2'].apply(lambda x: x if (x < 200) | np.isnan(x) else 200)

    df['f3'] = df['f3'].apply(lambda x: x if (x < 300) | np.isnan(x) else 300)

    df['f4'] = df['f4'].apply(lambda x: x if (x > 60) | np.isnan(x) else 60)
    df['f4'] = df['f4'].apply(lambda x: x if (x < 95) | np.isnan(x) else 95)

    df['f5'] = df['f5'].apply(lambda x: x if (x > 35) | np.isnan(x) else 35)
    df['f5'] = df['f5'].apply(lambda x: x if (x < 55) | np.isnan(x) else 55)

    df['f6'] = df['f6'].apply(lambda x: x if (x > 20) | np.isnan(x) else 20)
    df['f6'] = df['f6'].apply(lambda x: x if (x < 50) | np.isnan(x) else 50)

    df['f7'] = df['f7'].apply(lambda x: x if (x > 0.75) | np.isnan(x) else 0.75)
    df['f7'] = df['f7'].apply(lambda x: x if (x < 2.5) | np.isnan(x) else 2.5)

    df['f8'] = df['f8'].apply(lambda x: x if (x < 20) | np.isnan(x) else 20)

    df['f9'] = df['f9'].apply(lambda x: x if (x < 12.5) | np.isnan(x) else 12.5)

    df['f10'] = df['f10'].apply(lambda x: x if (x < 3) | np.isnan(x) else 3)

    df['f13'] = df['f13'].apply(lambda x: x if (x < 140) | np.isnan(x) else 140)

    df['f17'] = df['f17'].apply(lambda x: x if (x < 2.5) | np.isnan(x) else 2.5)

    df['f19'] = df['f19'].apply(lambda x: x if (x < 12.5) | np.isnan(x) else 12.5)

    df['f20'] = df['f20'].apply(lambda x: x if (x < 16) | np.isnan(x) else 16)

    df['f27'] = df['f27'].apply(lambda x: x if (x < 22) | np.isnan(x) else 22)

    df['f28'] = df['f28'].apply(lambda x: x if (x < 500) | np.isnan(x) else 500)

    df['f32'] = df['f32'].apply(lambda x: x if (x > 30) | np.isnan(x) else 30)

    df['f33'] = df['f33'].apply(lambda x: x if (x < 60) | np.isnan(x) else 60)
    return df


def search_min_level_all(data, start_column_index):
    columns = data.columns
    min_level_dict = {}
    for index in range(start_column_index, len(columns)):
        column_data = data.iloc[:, index]
        min_level = search_min_level(column_data)
        min_level_dict[column_data.name] = min_level
    return min_level_dict


def search_max_level_all(data, start_column_index):
    columns = data.columns
    max_level_dict = {}
    for index in range(start_column_index, len(columns)):
        column_data = data.iloc[:, index]
        max_level = search_max_level(column_data)
        max_level_dict[column_data.name] = max_level
    return max_level_dict


def search_min_level(data):
    data_non_nan = data[np.isnan(data) == False]
    min_value = data_non_nan.min()
    mean_value = data_non_nan.mean()
    step = (mean_value - min_value)/100
    min_scale = 1
    min_level = min_value
    while True:
        min_level_data = data_non_nan[data_non_nan >= min_level]
        if min_scale - len(min_level_data)/len(data_non_nan) > 0.03:
            step = step/2
            min_level = min_level - step
        elif len(min_level_data)/len(data_non_nan) < 0.96 or step < 0.001:
            break
        else:
            min_level = min_level + step
    if min_value/min_level >= 0.9:
        min_level = min_value
    print(data_non_nan.name + '  min_level:' + str(min_level) + '  scale:' + str(len(min_level_data)/len(data_non_nan)))
    return min_level


def search_max_level(data):
    data_non_nan = data[np.isnan(data) == False]
    max_value = data_non_nan.max()
    mean_value = data_non_nan.mean()
    step = (max_value - mean_value)/100
    max_scale = 1
    max_level = max_value
    while True:
        max_level_data = data_non_nan[data_non_nan <= max_level]
        if max_scale - len(max_level_data)/len(data_non_nan) > 0.03:
            step = step/2
            max_level = max_level + step
        elif len(max_level_data)/len(data_non_nan) < 0.96 or step < 0.00001:
            break
        else:
            max_level = max_level - step
    if max_level/max_value >= 0.9:
        max_level = max_value
    print(data_non_nan.name + '  max_level:' + str(max_level) + '  scale:' + str(len(max_level_data)/len(data_non_nan)))
    return max_level


def value_to_multi_class(x):
    if x < 6.1:
        return 0
    elif x >= 6.1 & x <7:
        return 1
    else:
        return 2


def logloss_to_class(data, class_level=0.5):
    return [np.math.ceil(x - class_level) for x in data]


def softmax_to_class(data, level=0.5):
    classes = []
    for index in range(len(data)):
        class_type = 0 if np.max(data[index]) < level else np.where(data[index] == np.max(data[index]))[0][0]
        classes.append(class_type)
    return classes
