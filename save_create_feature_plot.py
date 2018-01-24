import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engineering.rank_feature_majority import rank_feature_majority_all
from feature_engineering.segment_raw_data import segment_raw_data
from feature_engineering.rank_feature import rank_feature, rank_feature_by_max, rank_feature_count
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor, balance_k_fold_regressor
from utils import create_scale_feature, normalize_data_frame
from sampling.sample import separate_high_median_normal

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data = train_data.drop(['体检日期'], axis=1)
test_data = test_data.drop(['体检日期'], axis=1)

columns = train_data.columns
str_columns = ['sex', 'age'] + ['f' + str(p) for p in range(len(columns)-2)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

new_train_target = pd.concat([train_data, train_target], axis=1)

new_train_target = new_train_target.sort_values(by='Y').reset_index(drop=True)

new_train_data = new_train_target.iloc[:, :-1]
new_train_target = new_train_target.iloc[:, -1]

new_train_data, test_data = rank_feature_majority_all(new_train_data, test_data)

new_train_data = create_scale_feature(new_train_data)

for index in range(new_train_data.shape[1]):
    prefix = new_train_data.iloc[:, index].name
    print('save:', prefix)
    x = range(new_train_data.shape[0])
    plt.scatter(x, new_train_data.iloc[:, index], label=new_train_data.columns[index])
    plt.savefig('range/' + prefix + '_range.jpg', dpi=100)
    plt.close()
