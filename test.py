import pandas as pd
from dateutil.parser import parse

from model_selection.cv import k_fold_regressor
from model_selection.regressor_model_factory import RegressorModelFactory

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

# 默认情况 1.41 0.93677 0.73714 0.848803 0.953473

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0, '??':1})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data['体检日期'] = (pd.to_datetime(train_data['体检日期']) - parse('2016-10-09')).dt.days
test_data['体检日期'] = (pd.to_datetime(test_data['体检日期']) - parse('2016-10-09')).dt.days

columns = train_data.columns
str_columns = ['sex', 'age', 'date'] + ['f' + str(p) for p in range(len(columns)-3)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

lgb_y_valid, kf_lgb_mse, cv_indexs = \
     k_fold_regressor(train_data, train_target, train_data, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)

