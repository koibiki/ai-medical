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
from model_selection.cv import k_fold_regressor, balance_k_fold_regressor
from utils import create_scale_feature, fix_min, fix_min_all
from sampling.sample import separate_high_median_normal
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')

train_data = train.iloc[:, 1:-1]
train_target = train.iloc[:, -1]
# train_target_class = train_target.apply()
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data = train_data.drop(['体检日期'], axis=1)
test_data = test_data.drop(['体检日期'], axis=1)

columns = train_data.columns
str_columns = ['sex', 'age'] + ['f' + str(p) for p in range(len(columns)-2)]

train_data.columns = str_columns
test_data.columns = str_columns
train_target.name = 'Y'

scale_train_data = train_data

scale_train_target = pd.concat([scale_train_data, train_target], axis=1)

high, median, normal = separate_high_median_normal(scale_train_target)
print(high.shape)

all = scale_train_target.copy()
all.fillna(all.median(axis=0), inplace=True)
all = fix_min_all(all)
all = create_scale_feature(all)
all.fillna(all.median(axis=0), inplace=True)

high.fillna(high.median(axis=0), inplace=True)
high = fix_min_all(high)
high = create_scale_feature(high)
high.fillna(high.median(axis=0), inplace=True)

normal.fillna(normal.median(axis=0), inplace=True)
normal = fix_min_all(normal)
normal = create_scale_feature(normal)
normal.fillna(normal.median(axis=0), inplace=True)
print(str(high.shape) + '  ' + str(normal.shape))


def plot_fenbu(data, prefix):
    data_describe = data.describe()
    x = range(len(data))
    plt.scatter(x, data, label=high.columns[index])
    print('min:' + str(data_describe['min']) + ' 25%:' + str(data_describe['25%']) + ' median:' + str(data_describe['50%']) + ' 75%:' + str(data_describe['75%']) + ' max:' + str(data_describe['max']))
    print(prefix + ' mean/std :', data_describe['mean']/data_describe['std'])
    print('mean:' + str(data_describe['mean']) + '  std:' + str(data_describe['std']))
    plt.savefig('plot/' + prefix + '.jpg', dpi=100)
    plt.close()


def plot_k_mean(data, prefix):
    k_mean_data = pd.DataFrame(data)
    K = range(1, 10)
    mean_distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(k_mean_data)
        mean_distortions.append(sum(np.min(cdist(k_mean_data, kmeans.cluster_centers_, 'euclidean'), axis=1))/k_mean_data.shape[0])
    plt.plot(K, mean_distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Average Dispersion')
    plt.title('Selection k with the Elbow Method')
    plt.savefig('kmean/' + prefix + '.jpg', dpi=100)
    plt.close()


for index in range(len(scale_train_target.columns)):

    print('index:' + str(index))
    high_data = high.iloc[:, index]

    normal_data = normal.iloc[:, index]

    all_data = all.iloc[:, index]

    prefix = high.columns[index]
    print(prefix)

    plot_k_mean(all_data, 'origin/' + prefix + '_all_')
    plot_fenbu(all_data, 'origin/' + prefix + '_all_')

    plot_k_mean(high_data, 'origin/' + prefix + '_high_')
    plot_fenbu(high_data, 'origin/' + prefix + '_high_')

    plot_k_mean(normal_data, 'origin/' + prefix + '_normal_')
    plot_fenbu(normal_data, 'origin/' + prefix + '_normal_')
    if normal_data.median() != 0 or np.isnan(normal_data.median()) is False:
        print('median scale:', high_data.median()/normal_data.median())
    print('\t')


for index in range(len(scale_train_target.columns), len(high.columns)):

    print('index:' + str(index))
    high_data = high.iloc[:, index]

    normal_data = normal.iloc[:, index]

    all_data = all.iloc[:, index]

    prefix = high.columns[index]
    print(prefix)

    plot_k_mean(all_data, prefix + '_all_')
    plot_fenbu(all_data, prefix + '_all_')

    plot_k_mean(high_data, prefix + '_high_')
    plot_fenbu(high_data, prefix + '_high_')

    plot_k_mean(normal_data, prefix + '_normal_')
    plot_fenbu(normal_data, prefix + '_normal_')
    if normal_data.median() != 0 or np.isnan(normal_data.median()) is False:
        print('median scale:', high_data.median()/normal_data.median())
    print('\t')


