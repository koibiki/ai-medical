import pandas as pd

from utils import normalize_feature


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
            new_data.insert(new_data.shape[1], 'sum_' + columns[j] + '_' + columns[index], data.iloc[:, j] + data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]


def create_scale_feature(data):
    new_data = data.copy()
    columns = data.columns
    for index in range(0, len(columns)):
        for j in range(index + 1, len(columns)):
            new_data.insert(new_data.shape[1], 'scale_' + columns[j] + '_' + columns[index], data.iloc[:, j] * data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]


def create_subtract_feature(data):
    new_data = data.copy()
    columns = data.columns
    for index in range(0, len(columns)):
        for j in range(index + 1, len(columns)):
            new_data.insert(new_data.shape[1], 'subtract_' + columns[j] + '_' + columns[index], data.iloc[:, j] / data.iloc[:, index])
    return new_data.iloc[:, data.shape[1]:]
