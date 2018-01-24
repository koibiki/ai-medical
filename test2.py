import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from dateutil.parser import parse

from feature_engineering.nan_stastics import nan_statics
from feature_engineering.rank_feature_majority import rank_feature_majority_all, rank_feature_majority_train_valid_test
from feature_engineering.segment_raw_data import segment_raw_data
from feature_engineering.rank_feature import rank_feature, rank_feature_by_max, rank_feature_count
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor, balance_k_fold_regressor, create_sample_k_fold_regressor
from sampling.sample import sample_by_test_scale, separate_high_median_normal
from utils import create_scale_feature, normalize_data_frame, delete_error_data, filtration, create_sample

# 分别求出高血糖 中血糖 正常血糖每个feature 与其中位值的差值
# 将 血糖大于7的部分与 其他进行平衡抽样训练


train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')
# sample = pd.read_csv('input/d_sample_20180102.csv', ncoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
# train_target_class = train_target.apply()
test_data = test.iloc[:, 1:]
print(test_data.shape)

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

# train_data = train_data.drop(['体检日期'], axis=1)
# test_data = test_data.drop(['体检日期'], axis=1)
train_data['体检日期'] = (pd.to_datetime(train_data['体检日期']) - parse('2016-10-09')).dt.days
test_data['体检日期'] = (pd.to_datetime(test_data['体检日期']) - parse('2016-10-09')).dt.days

columns = train_data.columns
str_columns = ['sex', 'age', 'date'] + ['f' + str(p) for p in range(len(columns)-3)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

# train_data, test_data = nan_statics(train_data, test_data)

# train_data_target = pd.concat([train_data, train_target], axis=1)
#
# train_data_target = delete_error_data(train_data_target)
# train_data_target = filtration(train_data_target)
# test_data = filtration(test_data)

# train_data.fillna(-10, inplace=True)
# test_data.fillna(-10, inplace=True)


# train_data_target = pd.concat([train_data, train_target], axis=1)

# train, valid = sample_by_test_scale(train_data_target, test_data)


X_train, X_valid, y_train, y_valid = \
    train_test_split(train_data, train_target, test_size=0.1, random_state=33)

train_x_y = pd.concat([X_train, y_train], axis=1)

high, median, normal = separate_high_median_normal(train_x_y)
random_highs = [high]
for index in range(9):
    random_high = create_sample(high)
    random_highs.append(random_high)

new_train_x_y = pd.concat((random_highs + [median, normal]), axis=0)

X_train = new_train_x_y.iloc[:, :-1]
y_train = new_train_x_y.iloc[:, -1]

X_train, X_valid, test_data = rank_feature_majority_train_valid_test(X_train, X_valid, test_data)

drop_columns = str_columns[3:]
X_train = X_train.drop(drop_columns, axis=1)

X_valid = X_valid.drop(drop_columns, axis=1)


# X_train, X_valid, y_train, y_valid = train.iloc[:, :-1], valid.iloc[:, :-1], train.iloc[:, -1], valid.iloc[:, -1]


rmf = RegressorModelFactory()

lgb_y_valid, kf_lgb_mse, cv_indexs = \
    k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)
xgb_y_valid, kf_xgb_mse, cv_indexs = \
    k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_XGBOOST, cv=5)


y_pred = (lgb_y_valid + xgb_y_valid) / 2


print('kf mse:', (kf_lgb_mse + kf_xgb_mse)/(2 * 2))
print('mse : ', mean_squared_error(y_valid, y_pred)/2)

# pd_pred = pd.DataFrame(y_pred, columns=['血糖'])
# pd_pred.to_csv('output/prediction.csv', index=None, header=None)
# print(len(y_pred))
x = range(len(y_pred))
plt.plot(x, y_valid, 'r-*', label='y_valid')
plt.plot(x, y_pred, 'b-*', label='y_pred')
plt.legend()
plt.show()
