import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse

from feature_engineering.sum_value import create_sum_feature

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import normalize_data_frame

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

# 默认情况 1.41 0.93677 0.73714 0.848803 0.953473

# 默认情况 1.425

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0, '??':1})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data['体检日期'] = (pd.to_datetime(train_data['体检日期']) - parse('2016-10-09')).dt.days
test_data['体检日期'] = (pd.to_datetime(test_data['体检日期']) - parse('2016-10-09')).dt.days

columns = train_data.columns
str_columns = ['sex', 'age', 'date'] + ['f' + str(p) for p in range(len(columns)-3)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

# train_data, factors = normalize_data_frame(train_data)
#
# new_train_data = create_sum_feature(train_data)
#
# print(new_train_data.columns)
#
# train_data = pd.concat([train_data, new_train_data], axis=1)

print(train_data.columns)

X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.2, random_state=520)

model = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                          learning_rate=0.05, n_estimators=5000,
                          max_bin=55, bagging_fraction=0.8,
                          bagging_freq=5, feature_fraction=0.2319,
                          feature_fraction_seed=9, bagging_seed=9,
                          min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
model.fit(X_train, y_train)
print('mse:', mean_squared_error(y_valid, model.predict(X_valid)) * 0.5)

importance = pd.Series(model.feature_importances_, index=train_data.columns, name='importance')
importance = importance.sort_values(ascending=False)

print(importance)
important_feature = importance.iloc[:20]
print(important_feature.index.values)

X_train = X_train.loc[:, important_feature.index.values]
X_valid = X_valid.loc[:, important_feature.index.values]

model = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                          learning_rate=0.05, n_estimators=1000,
                          max_bin=55, bagging_fraction=0.8,
                          bagging_freq=5, feature_fraction=0.2319,
                          feature_fraction_seed=9, bagging_seed=9,
                          min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
model.fit(X_train, y_train)
print('mse:', mean_squared_error(y_valid, model.predict(X_valid)) * 0.5)


