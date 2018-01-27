import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from dateutil.parser import parse
from sklearn.metrics import classification_report

from imblearn.combine import SMOTETomek, SMOTEENN

from feature_engineering.nan_stastics import nan_statics
from feature_engineering.rank_feature_majority import rank_feature_majority_all, rank_feature_majority_train_valid_test
from feature_engineering.segment_raw_data import segment_raw_data
from feature_engineering.rank_feature import rank_feature, rank_feature_by_max, rank_feature_count
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from model_selection.multi_classifier_model_factory import MultiClassifierModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor, k_fold_classifier, create_sample_k_fold_regressor, \
    k_fold_multi_classifier, calculate_multi_mean
from sampling.sample import sample_by_test_scale, separate_high_median_normal, separate_high_normal
from utils import create_scale_feature, normalize_data_frame, delete_error_data, filtration, create_sample, logloss_to_class, softmax_to_class

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].apply(lambda x:1 if x == '男' else 0)
test_data['性别'] = test_data['性别'].apply(lambda x:1 if x == '男' else 0)

train_data['体检日期'] = (pd.to_datetime(train_data['体检日期']) - parse('2016-10-09')).dt.days
test_data['体检日期'] = (pd.to_datetime(test_data['体检日期']) - parse('2016-10-09')).dt.days

columns = train_data.columns
str_columns = ['sex', 'age', 'date'] + ['f' + str(p) for p in range(len(columns)-3)]


def value_to_multi_class(x):
    if x < 6.1:
        return 0
    elif (x >= 6.1) & (x < 6.5):
        return 1
    else:
        return 2


train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'
train_target_class = train_target.apply(lambda x: value_to_multi_class(x))
train_target_class.name = 'class'

train_test = pd.concat([train_data, test_data], axis=0)
train_test, factors = normalize_data_frame(train_test, start_index=2)
train_data = train_test.iloc[:train_data.shape[0]]
test_data = train_test.iloc[train_data.shape[0]:]

train_data.fillna(-99, inplace=True)
test_data.fillna(-99, inplace=True)

train_data_target = pd.concat([train_data, train_target], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(train_data_target, train_target_class, test_size=0.2, random_state=20)

sm = SMOTETomek()
X_resampled, y_resampled = sm.fit_sample(X_train, y_train.reshape(-1, 1))

X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
y_train = pd.Series(y_resampled, name='class')


X_train_data = X_train.iloc[:, :-1]
X_valid_data = X_valid.iloc[:, :-1]

print(X_valid_data.columns)
print(X_valid_data.shape)

lgb_y_valid = \
    k_fold_multi_classifier(X_train_data, y_train, X_valid_data, MultiClassifierModelFactory.MODEL_LIGHET_GBM, cv=5)


print(lgb_y_valid)
print(len(y_valid))
print(softmax_to_class(lgb_y_valid))
print(classification_report(y_valid, softmax_to_class(lgb_y_valid, level=0.9)))