# coding: utf-8
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from imblearn.under_sampling import RandomUnderSampler

sys.path.append('../')
from util.feature import add_feature, fillna
from util.metric import mse
from util import variables

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)

predictor = [column for column in train.columns if column not in ['id', '体检日期', '血糖']]
rus = RandomUnderSampler(random_state=2018, return_indices=True)
XALL, yALL, idx_resampled = rus.fit_sample(train[predictor], (train['血糖']>7).astype(int))
yALL = train.iloc[idx_resampled]['血糖']
XALL = pd.DataFrame(XALL, columns=predictor)
print('Feature: ', XALL.columns.tolist())

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
preds = np.zeros((train.shape[0], 5))
feature_importance = []
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_dat = lgb.Dataset(XALL.iloc[train_idx], yALL.iloc[train_idx])
    valid_dat = lgb.Dataset(XALL.iloc[valid_idx], yALL.iloc[valid_idx])

    gbm = lgb.train(variables.lgb_params,
                   train_dat,
                   num_boost_round=variables.num_boost_round,
                   valid_sets=valid_dat,
                   verbose_eval=100,
                   early_stopping_rounds=variables.early_stopping_rounds,
                   feval=mse)
    
    preds[:, cv_idx] = gbm.predict(train[predictor],
                                     num_iteration=gbm.best_iteration)
    feature_importance.append(pd.DataFrame(gbm.feature_importance(),
                                          index=predictor, columns=['CV{0}'.format(cv_idx)]))

print('Offline mse: {0}'.format(mean_squared_error(train['血糖'], preds.mean(axis=1))*0.5))
feature_importance = pd.concat(feature_importance, axis=1)
feature_importance.sort_values(by='CV0', ascending=True, inplace=True)
feature_importance.to_csv('../feature_importance/feature_importance_tree.csv')
# fig_imp, ax_imp = plt.subplots(figsize=(6, 9*XALL.shape[1]//40))
# feature_importance.plot.barh(ax=ax_imp)
# fig_imp.tight_layout()
# fig_imp.savefig('../feature_importance/feature_importance_tree.png', dpi=200)
# fig_imp.show()
