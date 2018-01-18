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
from model_selection.cv import k_fold_regressor, balance_k_fold_regressor, balance_k_fold_regressor2
from utils import create_scale_feature

# 求每个feature 与其中位值的差值
# 将 血糖大于7的部分与 其他进行平衡抽样训练

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')
# sample = pd.read_csv('input/d_sample_20180102.csv', ncoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
# train_target_class = train_target.apply()
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data = train_data.drop(['体检日期'], axis=1)
test_data = test_data.drop(['体检日期'], axis=1)

train_data.fillna(train_data.median(axis=0), inplace=True)
test_data.fillna(test_data.median(axis=0), inplace=True)

columns = train_data.columns
str_columns = ['sex', 'age'] + ['f' + str(p) for p in range(len(columns)-2)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

scale_train_data = train_data

# scale_train_data = create_scale_feature(train_data.iloc[:, 1:])
# scale_train_data = pd.concat([train_data['sex'], scale_train_data], axis=1)

new_feature1 = segment_raw_data(scale_train_data.iloc[:, 2], 2)
scale_train_data = pd.concat([new_feature1, scale_train_data], axis=1)

print(scale_train_data.shape)
print(scale_train_data.head())
# cmf = ClassifierModelFactory()
scale_train_target = pd.concat([scale_train_data, train_target], axis=1)

high_train_target = scale_train_target[scale_train_target['Y'] >= 7]
median_train_target = scale_train_target[(scale_train_target['Y'] >= 6.1).values & (scale_train_target['Y'] < 7).values]
normal_train_target = scale_train_target[scale_train_target['Y'] < 6.1]




rmf = RegressorModelFactory()

X_train, X_valid, y_train, y_valid = train_test_split(scale_train_data, train_target, test_size=0.1, random_state=33)

lgb_y_valid, kf_lgb_mse = balance_k_fold_regressor2(X_train, y_train, X_valid, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)
# xgb_y_valid, kf_xgb_mse = k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_XGBOOST, cv=5)

# model = rmf.create_model(RegressorModelFactory.MODEL_XGBOOST)
# model = cmf.create_model(ClassifierModelFactory.MODEL_XGBOOST)
# model.fit(X_train, X_valid, y_train, y_valid)
# predict = model.predict(X_valid)
# print(predict)
y_pred = (lgb_y_valid )
#
print('kf mse:', (kf_lgb_mse )/2)
print('mse : ', mean_squared_error(y_valid, y_pred)/2)
print(y_pred)

# pd_pred = pd.DataFrame(y_pred, columns=['血糖'])
# pd_pred.to_csv('output/prediction.csv', index=None, header=None)
# x = range(len(y_pred))
# plt.plot(x, y_pred, 'r-*', label='y_pred')
# plt.plot(x, y_valid, 'b-o', label='y_valid')
# plt.legend()
# plt.show()
