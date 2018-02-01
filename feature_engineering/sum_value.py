import pandas as pd
import numpy as np
import math


exclude_columns = ['date', 'sex', 'weekend']


def sum_all_columns(data):
    data_sex_age_date = data.iloc[:, :3]
    data_other = data.iloc[:, 3:]
    data_other['sum_all'] = data.sum(axis=1)
    data_other['sum_all'], factors = normalize_feature(data_other['sum_all'])
    return pd.concat([data_sex_age_date, data_other], axis=1)


def create_sum_feature(data):
    new_data = data.copy()
    columns = data.columns
    for index in range(0, len(columns)):
        for j in range(index + 1, len(columns)):
            if columns[j] not in exclude_columns and columns[index] not in exclude_columns:
                new_data.insert(new_data.shape[1], 'sum_' + columns[j] + '_' + columns[index],
                                data.iloc[:, j] + data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]


def create_scale_feature(data):
    new_data = data.copy()
    columns = data.columns.values
    for index in range(0, len(columns)):
        for j in range(index + 1, len(columns)):
            if columns[j] not in exclude_columns and columns[index] not in exclude_columns:
                new_data.insert(new_data.shape[1], 'scale_' + columns[j] + '_' + columns[index],
                                data.iloc[:, j] * data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]


def create_subtract_feature(data):
    new_data = data.copy()
    columns = data.columns.values
    for index in range(0, len(columns)):
        for j in range(index + 1, len(columns)):
            new_data.insert(new_data.shape[1], 'subtract_' + columns[j] + '_' + columns[index],
                            data.iloc[:, j] - data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]


def create_divide_feature(data):
    new_data = data.copy()
    columns = [column for column in data.columns.values if column not in ['date', 'sex']]
    for index in range(0, len(columns)):
        for j in range(index + 1, len(columns)):
            if columns[j] not in exclude_columns and columns[index] not in exclude_columns:
                new_data.insert(new_data.shape[1], 'divide_' + columns[j] + '_' + columns[index],
                                data.iloc[:, j] / data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]


def create_square_feature(data):
    new_data = data.copy()
    columns = [column for column in data.columns.values if column not in exclude_columns]
    for item in columns:
        new_data.insert(new_data.shape[1], 'square_' + item, data[item] * data[item])
    return new_data.iloc[:, data.shape[1]:]


def create_log_feature(data):
    new_data = data.copy()
    columns = [column for column in data.columns.values if column not in exclude_columns]
    for item in columns:
        new_data['log_' + item] = data[item].apply(lambda x: x if (type(x) != np.float64) else np.math.log10(x))
    return new_data.iloc[:, data.shape[1]:]


def create_extract_root_feature(data):
    columns = [column for column in data.columns.values if column not in exclude_columns]
    new_data = data.loc[:, columns]
    for item in columns:
        new_data['extract_root_' + item] = data[item].apply(lambda x: x if (type(x) != np.float64) else math.sqrt(x))
    return new_data.iloc[:, data.shape[1]:]


def create_mix_sum_feature(data1, data2):
    columns1 = [column for column in data1.columns.values if column not in exclude_columns]
    columns2 = [column for column in data2.columns.values if column not in exclude_columns]
    mix_sum_feature = None
    for column1 in columns1:
        for column2 in columns2:
            if mix_sum_feature is None:
                mix_sum_feature = pd.DataFrame(data1[column1] + data2[column2],
                                               columns=['sum_' + column1 + '_' + column2])
            else:
                mix_sum_feature['sum_' + column1 + '_' + column2] = data1[column1] + data2[column2]
    return mix_sum_feature


def normalize_feature(data, f_min=0, f_max=100):
    d_min, d_max = min(data), max(data)
    factor = (f_max - f_min) / (d_max - d_min)
    normalized = f_min + (data - d_min) * factor
    return normalized, factor


def normalize_data_frame(df):
    factors = {}
    columns = [column for column in df.columns if column not in (exclude_columns + ['age'])]
    for item in columns:
        df[item], factor = normalize_feature(data=df[item])
        factors[item] = factor
    return df, factors
