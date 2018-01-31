#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:38:15 2018

@author: wufei
"""

# Parameters

XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models



import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from dateutil.parser import parse
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from catboost import CatBoostRegressor
from tqdm import tqdm
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import time
import sklearn
###### READ IN RAW DATA
#
print( "\nReading data from disk ...")

data_path = '/home/wufei/data/DM/'

train0 = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb18030')
trainA = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb18030')
test = pd.read_csv(data_path+'d_test_B_20180128.csv',encoding='gb18030')
train=pd.concat([train0,trainA])


#train = pd.read_csv(data_path+'x_svmAtrain.csv',encoding='gb18030')
#test = pd.read_csv(data_path+'d_test_B_20180128.csv',encoding='gb18030')



#predictors columns
    
#predictors = [f for f in train.columns if f not in ['血糖','id']]
#
#predictors1 = [f for f in train.columns if f not in ['血糖','id']]

predictors = [f for f in train.columns if f not in ['血糖','id','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原','乙肝e抗体', '乙肝核心抗体']]

predictors1 = [f for f in train.columns if f not in ['血糖','id','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原','乙肝e抗体', '乙肝核心抗体']]


predictCol='血糖'






def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days
    data.fillna(data.median(axis=0),inplace=True)
#    data.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原','乙肝e抗体', '乙肝核心抗体'],axis=1)
    scaler = sklearn.preprocessing.MinMaxScaler()
    data[predictors1]= scaler.fit_transform(data[predictors1])
#    data.fillna(-999,axis=1,inplace=True)
   
    
    
    for c, dtype in zip(data.columns, data.dtypes):	
        if dtype == np.float64:
            data[c] = data[c].astype(np.float32)
    
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    
    return train_feat,test_feat



def MSE(y, ypred):
    return np.sum([np.square(y[i]-ypred[i])  for i in range(len(y))]) / (2*len(y))



train_feat,test_feat = make_feat(train,test)

print('开始CV 5折训练...')
#t0 = time.time()

train_preds = np.zeros((train_feat.shape[0],3))

test_preds_lgb = np.zeros((test_feat.shape[0], 5))
test_preds_xgb = np.zeros((test_feat.shape[0], 5))
test_preds_cat = np.zeros((test_feat.shape[0], 5))
test_preds_ols = np.zeros((test_feat.shape[0], 5))

scores_lgb = np.zeros(5)
scores_xgb = np.zeros(5)
scores_cat = np.zeros(5)
scores_ols = np.zeros(5)



kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)

for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    
    df_train = train_feat.iloc[train_index]
    df_test = train_feat.iloc[test_index]

#################
###  LightGBM  ##
#################
#################
#
##### PROCESS DATA FOR LIGHTGBM

    print( "\nProcessing data for LightGBM ..." )
    
    
#    df_train,df_test= make_feat(train,test)
    
    x_train = df_train[predictors]
    
    y_train = df_train[predictCol].values
    
    print(x_train.shape, y_train.shape)
    print("   Preparing x_test...")
    
    x_test = df_test[predictors]

    
    train_columns = x_train.columns
    
    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)
    
    
    x_train = x_train.values.astype(np.float32, copy=False)
    
    d_train = lgb.Dataset(x_train, label=y_train)
    d_test = lgb.Dataset(df_test[predictors],df_test[predictCol])
    
    def evalerror(pred, df):
        label = df.get_label().values.copy()
        score = mean_squared_error(label,pred)*0.5
        return ('0.5mse',score,False)
    
    ##### RUN LIGHTGBM
    
    params = {}
    params['max_bin'] = 20
    params['learning_rate'] = 0.0021 # shrinkage_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'mse'          # or 'mae'
    params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
    params['bagging_fraction'] = 0.85 # sub_row
    params['bagging_freq'] = 40
    params['num_leaves'] = 31        # num_leaf
    params['min_data'] = 200         # min_data_in_leaf
    params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    params['verbose'] = 0
    params['feature_fraction_seed'] = 2
    params['bagging_seed'] = 3
    
    np.random.seed(0)
    random.seed(0)
    
    print("\nFitting LightGBM model ...")
    clf = lgb.train(params, d_train,
                    num_boost_round=3000,
                    valid_sets=d_test,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    del d_train; gc.collect()
    del x_train; gc.collect()
    
    print("\nPrepare for LightGBM prediction ...")
    

    
    print("\nStart LightGBM prediction ...")
    train_preds[test_index,0] += clf.predict(x_test, num_iteration=clf.best_iteration)
    
    test_preds_lgb[:,i] = clf.predict(test_feat[predictors], num_iteration=clf.best_iteration)
    

    
    del x_test; gc.collect()
    
    
    
    
    
    ################
    ################
    ##  XGBoost   ##
    ################
    ################
    
    #### PROCESS DATA FOR XGBOOST
    
    print( "\nProcessing data for XGBoost ...")
    
    
    
    x_train = df_train[predictors]
    
    y_train = df_train[predictCol].values
    
    
    y_mean = np.mean(y_train)
    
    print(x_train.shape, y_train.shape)
    
    
    
    x_test = df_test[predictors]
    # shape        
    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
    
    
    ##### RUN XGBOOST
    
    print("\nSetting up data for XGBoost ...")
    # xgboost params
    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'lambda': 0.8,   
        'alpha': 0.4, 
        'base_score': y_mean,
        'silent': 1,
        'max_delta_step':1
    }
    
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
    
    num_boost_rounds = 250
    print("num_boost_rounds="+str(num_boost_rounds))
    
    # train model
    print( "\nTraining XGBoost ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    
    print( "\nPredicting with XGBoost ...")
    xgb_pred1 = model.predict(dtest)
    
    dtest_feat= xgb.DMatrix(test_feat[predictors])
    xgb_pred_treat1 = model.predict(dtest_feat)
    
    
    print( "\nFirst XGBoost predictions:" )
    print( pd.DataFrame(xgb_pred1).head() )
    
    
    
    ##### RUN XGBOOST AGAIN
    
    print("\nSetting up data for XGBoost ...")
    # xgboost params
    xgb_params = {
        'eta': 0.033,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean,
        'silent': 1,
        'max_delta_step':1
    }
    
    num_boost_rounds = 150
    print("num_boost_rounds="+str(num_boost_rounds))
    
    print( "\nTraining XGBoost again ...")
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    
    print( "\nPredicting with XGBoost again ...")
    xgb_pred2 = model.predict(dtest)
    
    dtest_feat= xgb.DMatrix(test_feat[predictors])
    xgb_pred_treat2 = model.predict(dtest_feat)
    
    print( "\nSecond XGBoost predictions:" )
    print( pd.DataFrame(xgb_pred2).head() )
    
    
    
    ##### COMBINE XGBOOST RESULTS
    train_preds[test_index,1] += XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
    
    test_preds_xgb[:,i]=XGB1_WEIGHT*xgb_pred_treat1  + (1-XGB1_WEIGHT)*xgb_pred_treat2 
    
    
    del x_train
    del x_test
    del dtest
    del dtrain
    del xgb_pred1
    del xgb_pred2 
    gc.collect()
      
    
    ################
    ################
    ## catboost   ##
    ################
    ################
    print( "\n\nProcessing data for catboost ...")
    
#    train_df,test_df = make_feat(train,test)
    
    X_train = df_train[predictors]
    
    y_train = df_train[predictCol].values
    
    X_test = df_test[predictors]
    print(X_train.shape, y_train.shape)
    
    num_ensembles = 5
    y_pred_cat = 0.0
    treat_cat=0.0
    for j in tqdm(range(num_ensembles)):
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.03,
            depth=6, l2_leaf_reg=3, 
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=j)
        model.fit(X_train, y_train)
        y_pred_cat += model.predict(X_test)     
        treat_cat += model.predict(test_feat[predictors])
    y_pred_cat /= num_ensembles
    treat_cat /=num_ensembles
    
    train_preds[test_index,2] =y_pred_cat
    test_preds_cat[:,i]=treat_cat
    
 
    gc.collect()
    


    ################
    ################
    ##    OLS     ##
    ################
    ################
    
#    np.random.seed(17)
#    random.seed(17)
#    
#    print( "\n\nProcessing data for OLS ...")
#    
#
#    y = df_train[predictCol].values
#    train = df_train[predictors]
#    
#    test = df_test[predictors]
#    
#    # shape        
#    print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
#    
#    
#    exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] 
#    col = [c for c in train.columns if c not in exc]
#    
#    
#    
#    print("\nFitting OLS...")
#    reg = LinearRegression(n_jobs=-1)
#    reg.fit(train, y); print('fit...')
  
    
 #memory


########################
########################
##  Combine and Save  ##
########################
########################
FUDGE_FACTOR = 1.000 # Multiply forecasts by this

XGB_WEIGHT = 0.3200
BASELINE_WEIGHT = 0.0000
OLS_WEIGHT = 0.0000
CAT_WEIGHT=0.3300
BASELINE_PRED = 6.6
##### COMBINE PREDICTIONS
 
print( "\nCombining XGBoost, LightGBM and baseline predicitons ..." )
lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT - OLS_WEIGHT -CAT_WEIGHT 

lgb_weight0 = lgb_weight / (1 - OLS_WEIGHT)

xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)

cat_weight0= CAT_WEIGHT / (1 - OLS_WEIGHT)

baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
 
 
p_test1 = train_preds[:,0]
xgb_pred1=train_preds[:,1]
y_pred_cat1= train_preds[:,2]


pred1 = 0
pred1 += xgb_weight0*xgb_pred1
pred1 += baseline_weight0*BASELINE_PRED
pred1 += lgb_weight0*p_test1
pred1 += cat_weight0*y_pred_cat1
#print( "\nCombined XGB/LGB/NN/CAT/baseline predictions:" )
#print( pd.DataFrame(pred1).head() )

#print( "\nPredicting with OLS and combining with XGB/LGB/NN/CAT/baseline predicitons: ..." )
pred_train = FUDGE_FACTOR * ( (1-OLS_WEIGHT)*pred1 )

  
#pred_train = FUDGE_FACTOR * ( OLS_WEIGHT*reg.predict(train_feat[predictors]) + (1-OLS_WEIGHT)*pred1 )

print('result score-------------')

print(MSE(pred_train,train_feat[predictCol].values))

 

xgb_pred=test_preds_xgb.mean(axis=1)
y_pred_cat=test_preds_cat.mean(axis=1)
p_test=    test_preds_lgb.mean(axis=1)


pred0 = 0
pred0 += xgb_weight0*xgb_pred
pred0 += baseline_weight0*BASELINE_PRED
pred0 += lgb_weight0*p_test
pred0 += cat_weight0*y_pred_cat
#print( "\nCombined XGB/LGB/NN/CAT/baseline predictions:" )


#print( "\nPredicting with OLS and combining with XGB/LGB/NN/CAT/baseline predicitons: ..." )
pred = FUDGE_FACTOR * ((1-OLS_WEIGHT)*pred0 )

  
#pred = FUDGE_FACTOR * ( OLS_WEIGHT*reg.predict(test_feat[predictors]) + (1-OLS_WEIGHT)*pred0 )


submission = [float(format(x, '.4f')) for x in pred]


submission=pd.DataFrame(submission)
    

##### WRITE THE RESULTS

from datetime import datetime

print( "\nWriting results to disk ..." )
submission.to_csv('subA{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,header=False)

print( "\nFinished ...")



