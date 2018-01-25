import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from dateutil.parser import parse
from sklearn.metrics import classification_report

from imblearn.combine import SMOTETomek, SMOTEENN

from feature_engineering.nan_stastics import nan_statics
from feature_engineering.rank_feature_majority import rank_feature_majority_all, rank_feature_majority_train_valid_test
from feature_engineering.segment_raw_data import segment_raw_data
from feature_engineering.rank_feature import rank_feature, rank_feature_by_max, rank_feature_count
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from model_selection.multi_classifier_model_factory import MultiClassifierModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from model_selection.cv import k_fold_regressor, k_fold_classifier, create_sample_k_fold_regressor
from sampling.sample import sample_by_test_scale, separate_high_median_normal, separate_high_normal
from utils import create_scale_feature, normalize_data_frame, delete_error_data, filtration, create_sample, logloss_to_class, softmax_to_class

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].apply(lambda x:1 if x == '男' else 0)
test_data['性别'] = test_data['性别'].apply(lambda x:1 if x == '男' else 0)

train_data['体检日期'] = (pd.to_datetime(train_data['体检日期']) - parse('2016-10-09')).dt.days
test_data['体检日期'] = (pd.to_datetime(test_data['体检日期']) - parse('2016-10-09')).dt.days

columns = train_data.columns
str_columns = ['sex', 'age', 'date'] + ['f' + str(p) for p in range(len(columns)-3)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'
train_target_class = train_target.apply(lambda x: 1 if x > 7 else 0)

train_test = pd.concat([train_data, test_data], axis=0)
train_test, factors = normalize_data_frame(train_test, start_index=2)
train_data = train_test.iloc[:train_data.shape[0]]
test_data = train_test.iloc[train_data.shape[0]:]

train_data.fillna(-99, inplace=True)
test_data.fillna(-99, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target_class, test_size=0.2, random_state=20)

sm = SMOTETomek()
X_resampled, y_resampled = sm.fit_sample(X_train, y_train.reshape(-1, 1))

X_train = pd.DataFrame(X_resampled, columns= X_train.columns)
y_train = pd.Series(y_resampled, name='Y')

lgb_y_valid, kf_lgb_mse = \
    k_fold_classifier(X_train, y_train, X_valid, ClassifierModelFactory.MODEL_LIGHET_GBM, cv=5)

y_pred = logloss_to_class(lgb_y_valid, class_level=0.7)

print(classification_report(y_valid, y_pred))

valid = pd.Series(y_valid, name='valid').reset_index(drop=True)
pred = pd.Series(y_pred, name='pred').reset_index(drop=True)

df = pd.DataFrame(valid)
df['pred'] = pred

print('valid:1  pred:0  ', len(df[(df['valid']==1).values & (df['pred']==0).values]))
print('valid:0  pred:1  ', len(df[(df['valid']==0).values & (df['pred']==1).values]))
print('valid:1  pred:1  ', len(df[(df['valid']==1).values & (df['pred']==1).values]))