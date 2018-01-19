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
from utils import create_scale_feature
from sampling.sample import separate_high_median_normal

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
# train_target_class = train_target.apply()
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

scale_train_data = train_data

scale_train_target = pd.concat([scale_train_data, train_target], axis=1)

high, median, normal = separate_high_median_normal(scale_train_target)

high_describe = high.describe()
median_describe = median.describe()
normal_describe = normal.describe()
high_median = high_describe.median()
median_median = median_describe.median()
normal_median = normal_describe.median()

for index in range(len(scale_train_target.columns)):
    high_data = high.iloc[:, index]
    high_data_describe = high_data.describe()
    normal_data = normal.iloc[:, index]
    normal_data_describe = normal_data.describe()

    prefix = scale_train_target.columns[index]
    print(prefix)

    x = range(len(high_data))
    plt.scatter(x, high_data, label=scale_train_target.columns[index])
    print('min:' + str(high_data.min()) + ' 25%:' + str(high_data_describe['25%']) + ' median:' + str(high_data.median()) + ' 75%:' + str(high_data_describe['75%']) + ' max:' + str(high_data.max()))
    print('high mean/std :', high_data_describe['mean']/high_data_describe['std'])
    print('mean:' + str(high_data_describe['mean']) + '  std:' + str(high_data_describe['std']))
    plt.savefig('plot/' + prefix + '_high.jpg', dpi=100)
    plt.close()

    x = range(len(normal_data))
    plt.scatter(x, normal_data, label=scale_train_target.columns[index])
    print('min:' + str(normal_data.min()) + ' 25%:' + str(normal_data_describe['25%']) + ' median:' + str(normal_data.median()) + ' 75%:' + str(normal_data_describe['75%']) + ' max:' + str(normal_data.max()))
    print('normal mean/std :', normal_data_describe['mean']/normal_data_describe['std'])
    print('mean:' + str(normal_data_describe['mean']) + '  std:' + str(normal_data_describe['std']))
    plt.savefig('plot/' + prefix + '_normal.jpg', dpi=100)
    plt.close()
    if normal_data.median() != 0 or np.isnan(normal_data.median()) is False:
        print('median scale:', high_data.median()/normal_data.median())
    print('\t')
