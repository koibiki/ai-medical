import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from dateutil.parser import parse

from feature_engineering.nan_stastics import nan_statics
from feature_engineering.segment_raw_data import segment_raw_data
from feature_engineering.rank_feature import rank_feature, rank_feature_by_max, rank_feature_count
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor, balance_k_fold_regressor
from sampling.sample import sample_by_test_scale
from utils import create_scale_feature, normalize_data_frame, delete_error_data, filtration

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

# date = pd.get_dummies(train_data['date'])
# train_data = train_data.drop(['date'], axis=1)
# train_data = pd.concat([train_data, date], axis=1)

train_data_target = pd.concat([train_data, train_target], axis=1)

train_data_target = delete_error_data(train_data_target)
train_data_target = filtration(train_data_target)
test_data = filtration(test_data)

train_data = train_data_target.iloc[:, 1:-1]
train_target = train_data_target.iloc[:, -1]

train_test = pd.concat([train_data, test_data], axis=0)

train_test, factors = normalize_data_frame(train_test, start_index=2)
train_data = train_test.iloc[:train_data.shape[0]]
test_data = train_test.iloc[train_data.shape[0]:]

train_data.fillna(-1, inplace=True)
test_data.fillna(-1, inplace=True)

train_data_target = pd.concat([train_data, train_target], axis=1)

train, valid = sample_by_test_scale(train_data_target, test_data)
print(train.shape)
print(valid.shape)
# scale_train_data = train_data
# scale_train_data = create_scale_feature(train_data.iloc[:, 1:])
# scale_train_data = pd.concat([train_data['sex'], scale_train_data], axis=1)

# new_feature1 = segment_raw_data(scale_train_data.iloc[:, 2], 2)
# scale_train_data = pd.concat([new_feature1, scale_train_data], axis=1)

# cmf = ClassifierModelFactory()
# scale_train_target = pd.concat([scale_train_data, train_target], axis=1)


rmf = RegressorModelFactory()

# X_train, X_valid, y_train, y_valid = \
#     train_test_split(train_data, train_target, test_size=0.1, random_state=33)

X_train, X_valid, y_train, y_valid = train.iloc[:, :-1], valid.iloc[:, :-1], train.iloc[:, -1], valid.iloc[:, -1]
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
# lgb_y_valid, kf_lgb_mse, cv_indexs = \
#      k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)
# xgb_y_valid, kf_xgb_mse, cv_indexs = \
#     k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)
y_preds = []
mses = []
for index in range(14):
    xgb_y_valid, kf_xgb_mse, cv_indexs = k_fold_regressor(X_train, y_train, X_valid, index, cv=5)
    y_preds.append(xgb_y_valid)
    mses.append(kf_xgb_mse)

sum = y_preds[0]
for index in range(1, 14):
    sum += y_preds[index]


# model = rmf.create_model(RegressorModelFactory.MODEL_XGBOOST)
# model = cmf.create_model(ClassifierModelFactory.MODEL_XGBOOST)
# model.fit(X_train, X_valid, y_train, y_valid)
# predict = model.predict(X_valid)
# print(predict)
y_pred = sum/len(y_preds)

#
print('kf mse:', np.sum(mses)/(2 * len(mses)))
print('mse : ', mean_squared_error(y_valid, y_pred)/2)

# pd_pred = pd.DataFrame(y_pred, columns=['血糖'])
# pd_pred.to_csv('output/prediction.csv', index=None, header=None)
# print(len(y_pred))
x = range(len(y_pred))
plt.plot(x, y_pred - y_valid, 'r-*', label='y_pred')
plt.legend()
plt.show()
