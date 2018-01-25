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


# 需要在修改时间格式后运行
def sample_by_test_scale(train, test):
    train_date = train['date'].unique()
    test_date = test['date'].unique()
    valid_set = []
    train_set = []
    for date in test_date:
        test_date_item = test[test['date'] == date]
        train_date_item = train[train['date'] == date]
        scale = test_date_item.shape[0]/train_date_item.shape[0]
        print('date:' + str(date) + '  scale:' + str(scale))
        X_train, X_valid, y_train, y_valid = \
            train_test_split(train_date_item.iloc[:, :-1], train_date_item.iloc[:, -1], test_size=scale, random_state=33)
        valid_item = pd.concat([X_valid, y_valid], axis=1)
        train_item = pd.concat([X_train, y_train], axis=1)
        valid_set.append(valid_item)
        train_set.append(train_item)
    train_dates = [date for date in train_date if date not in test_date]
    for date in train_dates:
        train_set.append(train[train['date'] == date])
    train_all = pd.concat(train_set, axis=0)
    valid_all = pd.concat(valid_set, axis=0)
    return train_all, valid_all


def separate_high_median_normal(data):
    high = data[data['Y'] >= 7]
    median = data[(data['Y'] >= 6.1).values & (data['Y'] < 7).values]
    normal = data[data['Y'] < 6.1]
    return high, median, normal


def separate_high_normal(data):
    return data[data.iloc[:, -1] == 1], data[data.iloc[:, -1] == 0]
