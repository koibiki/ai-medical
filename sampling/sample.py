import pandas as pd
from sklearn.model_selection import train_test_split


def balance_sample(data):
    high, median, normal = separate_high_median_normal(data)

    X_train_high, X_valid_high, y_train_high, y_valid_high = \
        train_test_split(high.iloc[:, :-1], high.iloc[:, -1], test_size=0.1, random_state=33)
    X_train_median, X_valid_median, y_train_median, y_valid_median = \
        train_test_split(median.iloc[:, :-1], median.iloc[:, -1], test_size=0.1, random_state=33)
    X_train_normal, X_valid_normal, y_train_normal, y_valid_normal = \
        train_test_split(normal.iloc[:, :-1], normal.iloc[:, -1], test_size=0.1, random_state=33)

    X_train = pd.concat([X_train_high, X_train_median, X_train_normal], axis=0).reset_index(drop=True)
    X_valid = pd.concat([X_valid_high, X_valid_median, X_valid_normal], axis=0).reset_index(drop=True)
    y_train = pd.concat([y_train_high, y_train_median, y_train_normal], axis=0).reset_index(drop=True)
    y_valid = pd.concat([y_valid_high, y_valid_median, y_valid_normal], axis=0).reset_index(drop=True)
    return X_train, X_valid, y_train, y_valid


def separate_high_median_normal(data):
    high = data[data['Y'] >= 7]
    median = data[(data['Y'] >= 6.1).values & (data['Y'] < 7).values]
    normal = data[data['Y'] < 6.1]
    return high, median, normal
