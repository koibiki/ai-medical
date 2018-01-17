import random
import pandas as pd


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
        for j in range(index + 1, len(columns)):
            new_data.insert(new_data.shape[1], columns[j] + '_' + columns[index], data.iloc[:, j]/data.iloc[:, index])
    return new_data
