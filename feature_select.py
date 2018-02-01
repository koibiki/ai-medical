import pandas as pd

from feature_engineering.sum_value import create_sum_feature, create_scale_feature, create_subtract_feature, \
    create_divide_feature, create_square_feature, create_log_feature, create_extract_root_feature, normalize_data_frame
from model_selection.cv import k_fold_regressor

from model_selection.regressor_model_factory import RegressorModelFactory

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

# 默认情况 1.41 0.93677 0.73714 0.848803 0.953473

# 默认情况 1.4159

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0, '??': 1})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data['体检日期'] = pd.to_datetime(train_data['体检日期'])
test_data['体检日期'] = pd.to_datetime(test_data['体检日期'])

train_data['weekend'] = train_data['体检日期'].apply(lambda x: 1 if((x.dayofweek == 6) | (x.dayofweek == 0)) else 0)
test_data['weekend'] = test_data['体检日期'].apply(lambda x: 1 if((x.dayofweek == 6) | (x.dayofweek == 0)) else 0)

train_data = train_data.drop(['体检日期'], axis=1)
test_data = test_data.drop(['体检日期'], axis=1)

columns = train_data.columns
str_columns = ['sex', 'age'] + ['f' + str(p) for p in range(len(columns)-2)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

train_data, factors = normalize_data_frame(train_data)

sum_feature = create_sum_feature(train_data)

scale_feature = create_scale_feature(train_data)

subtract_feature = create_subtract_feature(train_data)

train_data = pd.concat([train_data, sum_feature, scale_feature, subtract_feature], axis=1)

sum_names = pd.read_csv('output/2.csv', names=['sum'])['sum'].values
subtract_names = pd.read_csv('output/3.csv', names=['subtract'])['subtract'].values
scale_names = pd.read_csv('output/4.csv', names=['scale'])['scale'].values
print(sum_names)
print(str_columns)
print(train_data.columns)
train_columns = list(str_columns) + list(sum_names) + list(subtract_names) + list(scale_names)
print(train_columns)
train_data = train_data.loc[:, train_columns]

print(train_data.columns)
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.2, random_state=520)


# lgb_y_valid, kf_lgb_mse, cv_indexs = \
#     k_fold_regressor(train_data, train_target, train_data, RegressorModelFactory.MODEL_LIGHET_GBM, cv=5)


rmf = RegressorModelFactory()

model = rmf.create_model(RegressorModelFactory.MODEL_LIGHET_GBM)
model.fit(X_train, X_valid, y_train, y_valid)
print('mse:', mean_squared_error(y_valid, model.predict(X_valid)) * 0.5)


print(model.feature_importance())
importance = pd.Series(model.feature_importance(), index=X_train.columns, name='importance')
importance = importance.sort_values(ascending=False)

print(importance)
important_feature = importance.iloc[:200]
print(important_feature.index.values)

new_important_features = [item for item in important_feature.index.values if item not in str_columns]

print(new_important_features)

X_train_important = X_train.loc[:, new_important_features + str_columns]
X_valid_important = X_valid.loc[:, new_important_features + str_columns]

print(X_train_important.columns)
model = rmf.create_model(RegressorModelFactory.MODEL_LIGHET_GBM)
model.fit(X_train_important, X_valid_important, y_train, y_valid)
print('mse:', mean_squared_error(y_valid, model.predict(X_valid_important)) * 0.5)

print(model.feature_importance())
importance = pd.Series(model.feature_importance(), index=X_train_important.columns, name='importance')
importance = importance.sort_values(ascending=False)

print(importance)
new_important_features = [item for item in important_feature.index.values if item not in str_columns]
# pd_feature = pd.DataFrame(new_important_features, columns=['importance'])
# pd_feature.to_csv('output/5.csv', header=None, index=None)