import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engineering.segment_raw_data import segment_raw_data
from feature_engineering.rank_feature import rank_feature, rank_feature_by_max, rank_feature_count
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor, balance_k_fold_regressor
from utils import create_scale_feature, fix_min, fix_min_all
from sampling.sample import separate_high_median_normal
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from dateutil.parser import parse

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
# train_target_class = train_target.apply()
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data['体检日期'] = (pd.to_datetime(train_data['体检日期']) - parse('2016-10-09')).dt.days
train_data = train_data.drop(['体检日期'], axis=1)
test_data = test_data.drop(['体检日期'], axis=1)

columns = train_data.columns
str_columns = ['sex', 'age'] + ['f' + str(p) for p in range(len(columns)-2)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

scale_train_data = train_data

scale_train_target = pd.concat([scale_train_data, train_target], axis=1)
scale_train_target = scale_train_target.drop(['f15', 'f16', 'f17', 'f18', 'f19'], axis=1)

print(scale_train_target.shape)
print(scale_train_target.columns)
# scale_train_target = scale_train_target[scale_train_target['f1'] < 300]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f2'] < 200]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f3'] < 400]
print(scale_train_target.shape)
scale_train_target = scale_train_target[(scale_train_target['f4'] < 95).values & (scale_train_target['f4'] > 60).values]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f5'] > 35]
print(scale_train_target.shape)
scale_train_target = scale_train_target[(scale_train_target['f6'] < 60).values & (scale_train_target['f6'] > 10).values]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f7'] < 3]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f8'] < 20]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f9'] < 15]
print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f10'] < 4]
print(scale_train_target.shape)
# scale_train_target = scale_train_target[scale_train_target['f17'] < 10]
# print(scale_train_target.shape)
scale_train_target = scale_train_target[scale_train_target['f20'] < 17.5]
print(scale_train_target.shape)


high, median, normal = separate_high_median_normal(scale_train_target)



