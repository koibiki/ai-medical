import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory

cmf = ClassifierModelFactory()
rmf = RegressorModelFactory()


def k_fold_classifier(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV 5折训练...')
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    mses = []
    test_y_preds = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = cmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
        kf_X_valid = train_x.iloc[test_index]
        kf_y_valid = train_y.iloc[test_index]
        model.fit(kf_X_train, kf_y_train, kf_X_valid, kf_y_valid)
        kf_y_pred = model.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)
    print(cmf.get_model_name(model_num) + ' k fold validation:', sum(mses) / len(mses))
    predict = calculate_mean(test_y_preds)
    save_prediction(predict, model_num)
    return predict


def k_fold_regressor(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV 5折训练...')
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    mses = []
    test_y_preds = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = rmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
        kf_X_valid = train_x.iloc[test_index]
        kf_y_valid = train_y.iloc[test_index]
        model.fit(kf_X_train, kf_X_valid, kf_y_train, kf_y_valid)
        kf_y_pred = model.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)
    print(cmf.get_model_name(model_num) + ' k fold validation:', sum(mses) * 0.5 / len(mses))
    predict = calculate_mean(test_y_preds)
    save_prediction(predict, model_num)
    return predict, sum(mses)/len(mses)


def calculate_mean(preds):
    sum_pred = np.zeros(len(preds[0]))
    for item in preds:
        sum_pred += np.array(item)
    return sum_pred/len(preds)


def save_prediction(pred, model_num):
    pass