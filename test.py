import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor


columns = ['id', 'sex', 'age']

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')
# sample = pd.read_csv('input/d_sample_20180102.csv', ncoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data = train_data.drop(['体检日期'], axis=1)
test_data = test_data.drop(['体检日期'], axis=1)

train_data['高血糖'] = train_target.map(lambda x: 1 if x >= 8 else 0)
train_data.fillna(train_data.median(axis=0), inplace=True)
test_data.fillna(test_data.median(axis=0), inplace=True)

cmf = ClassifierModelFactory()
rmf = RegressorModelFactory()

X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.1, random_state=33)

lgb_y_valid, kf_lgb_mse = k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)
xgb_y_valid, kf_xgb_mse = k_fold_regressor(X_train, y_train, X_valid, RegressorModelFactory.MODEL_XGBOOST, cv=5)

# model = rmf.create_model(RegressorModelFactory.MODEL_XGBOOST)
# model = cmf.create_model(ClassifierModelFactory.MODEL_XGBOOST)
# model.fit(X_train, X_valid, y_train, y_valid)
# predict = model.predict(X_valid)
# print(predict)
y_pred = (lgb_y_valid + xgb_y_valid)/2

print('kf mse:', (kf_lgb_mse + kf_xgb_mse)/4)
print('mse : ', mean_squared_error(y_valid, y_pred)/2)

# pd_pred = pd.DataFrame(y_pred, columns=['血糖'])
# pd_pred.to_csv('output/prediction.csv', index=None, header=None)
x = range(len(y_pred))
plt.plot(x, y_pred, 'r-*', label='y_pred')
plt.plot(x, y_valid, 'b-o', label='y_valid')
plt.legend()
plt.show()
