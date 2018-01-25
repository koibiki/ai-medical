import numpy as np
import pandas as pd

from sampling.sample import separate_high_median_normal

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from model_selection.multi_classifier_model_factory import MultiClassifierModelFactory
from model_selection.classifier_model_factory import ClassifierModelFactory
from model_selection.regressor_model_factory import RegressorModelFactory
from utils import create_sample, logloss_to_class, softmax_to_class

mcmf = MultiClassifierModelFactory()
cmf = ClassifierModelFactory()
rmf = RegressorModelFactory()


def k_fold_classifier(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV 5折训练...')
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    ass = []
    test_y_preds = []
    cv_indexs = {}
    for i, (train_index, test_index) in enumerate(kf.split(train_x)):
        print('第{}次训练...'.format(i))
        model = cmf.create_model(model_num)
        kf_X_train = train_x.iloc[train_index]
        kf_y_train = train_y.iloc[train_index]
        kf_X_valid = train_x.iloc[test_index]
        kf_y_valid = train_y.iloc[test_index]
        model.fit(kf_X_train, kf_X_valid, kf_y_train, kf_y_valid)
        kf_y_pred = model.predict(kf_X_valid)
        ass.append(accuracy_score(kf_y_valid, logloss_to_class(kf_y_pred)))
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)
        cv_indexs[i] = [train_index, test_index]
    print(cmf.get_model_name(model_num) + ' k fold validation:', sum(ass) / len(ass))
    predict = calculate_mean(test_y_preds)
    return predict, sum(ass) / len(ass)


def balance_k_fold_regressor(train_x, train_y, test_x, model_num, cv=5):
    print('开始balance2 CV 5折训练...')

    train_x_y = pd.concat([train_x, train_y], axis=1)
    high, median, normal = separate_high_median_normal(train_x_y)

    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    mses = []
    test_y_preds = []
    high_train_index = {}
    median_train_index = {}
    normal_train_index = {}
    for i, (train_index, test_index) in enumerate(kf.split(high)):
        high_train_index[i] = [train_index, test_index]

    for i, (train_index, test_index) in enumerate(kf.split(median)):
        median_train_index[i] = [train_index, test_index]

    for i, (train_index, test_index) in enumerate(kf.split(normal)):
        normal_train_index[i] = [train_index, test_index]

    for i in range(cv):
        print('第{}次训练...'.format(i))
        kf_high_X_train = high.iloc[high_train_index[i][0], :-1]
        kf_high_y_train = high.iloc[high_train_index[i][0], -1]
        kf_high_X_valid = high.iloc[high_train_index[i][1], :-1]
        kf_high_y_valid = high.iloc[high_train_index[i][1], -1]

        kf_median_X_train = median.iloc[median_train_index[i][0], :-1]
        kf_median_y_train = median.iloc[median_train_index[i][0], -1]
        kf_median_X_valid = median.iloc[median_train_index[i][1], :-1]
        kf_median_y_valid = median.iloc[median_train_index[i][1], -1]

        kf_normal_X_train = normal.iloc[normal_train_index[i][0], :-1]
        kf_normal_y_train = normal.iloc[normal_train_index[i][0], -1]
        kf_normal_X_valid = normal.iloc[normal_train_index[i][1], :-1]
        kf_normal_y_valid = normal.iloc[normal_train_index[i][1], -1]


        # kf_normal_X_train, kf_normal_X_valid, kf_normal_y_train, kf_normal_y_valid = \
        #     train_test_split(kf_normal_X, kf_normal_y, test_size=0.2, random_state=33)

        model = rmf.create_model(model_num)
        kf_X_train = pd.concat([kf_high_X_train, kf_median_X_train, kf_normal_X_train], axis=0)
        kf_y_train = pd.concat([kf_high_y_train, kf_median_y_train, kf_normal_y_train], axis=0)
        kf_X_valid = pd.concat([kf_high_X_valid, kf_median_X_valid, kf_normal_X_valid], axis=0)
        kf_y_valid = pd.concat([kf_high_y_valid, kf_median_y_valid, kf_normal_y_valid], axis=0)
        model.fit(kf_X_train, kf_X_valid, kf_y_train, kf_y_valid)
        kf_y_pred = model.predict(kf_X_valid)
        mses.append(mean_squared_error(kf_y_pred, kf_y_valid))
        test_y_pred = model.predict(test_x)
        test_y_preds.append(test_y_pred)

    print(cmf.get_model_name(model_num) + ' k fold validation:', sum(mses) * 0.5 / len(mses))
    predict = calculate_mean(test_y_preds)
    return predict, sum(mses)/len(mses), high_train_index, median_train_index, normal_train_index


def create_sample_k_fold_regressor(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV 5折训练...')

    train_x_y = pd.concat([train_x, train_y], axis=1)
    high, median, normal = separate_high_median_normal(train_x_y)
    random_highs = [high]
    for index in range(9):
        random_high = create_sample(high)
        random_highs.append(random_high)

    new_train_x_y = pd.concat((random_highs + [normal]), axis=0)
    train_x = new_train_x_y.iloc[:, :-1]
    train_y = new_train_x_y.iloc[:, -1]

    print('train_x:' + str(train_x.shape))
    print('train_y:' + str(train_y.shape))

    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    mses = []
    test_y_preds = []
    cv_indexs ={}
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
        cv_indexs[i] = [train_index, test_index]
    print(rmf.get_model_name(model_num) + ' k fold validation:', sum(mses) * 0.5 / len(mses))
    predict = calculate_mean(test_y_preds)
    return predict, sum(mses)/len(mses), cv_indexs


def k_fold_regressor(train_x, train_y, test_x, model_num, cv=5):
    print('开始CV 5折训练...')
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    mses = []
    test_y_preds = []
    cv_indexs ={}
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
        cv_indexs[i] = [train_index, test_index]
    print(rmf.get_model_name(model_num) + ' k fold validation:', sum(mses) * 0.5 / len(mses))
    predict = calculate_mean(test_y_preds)
    return predict, sum(mses)/len(mses), cv_indexs


def calculate_mean(preds):
    sum_pred = np.zeros(len(preds[0]))
    for item in preds:
        sum_pred += np.array(item)
    return sum_pred/len(preds)
