import numpy as np
import pandas as pd

from model_selection.classifier_model_factory import ClassifierModelFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


columns = ['id', 'sex', 'age']

train = pd.read_csv('input/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('input/d_test_A_20180102.csv', encoding='gb2312')
# sample = pd.read_csv('input/d_sample_20180102.csv', ncoding='gb2312')

train_data = train.iloc[:, 1:]
train_target = train.iloc[:, -1]
test_data = test.iloc[:, 1:]

train_data['性别'] = train_data['性别'].map({'男': 1, '女': 0})
train_data['高血糖'] = train_data['血糖'].map(lambda x: 1 if x >= 8 else 0)
test_data['性别'] = test_data['性别'].map({'男': 1, '女': 0})

train_data = train_data.drop(['体检日期'], axis=1)

train_class_x = train_data.iloc[:, :-2]
train_class_y = train_data.iloc[:, -1]

X_train, X_valid, y_train, y_valid = train_test_split(train_class_x, train_class_y, test_size=0.25, random_state=33)

cmf = ClassifierModelFactory()
model = cmf.create_model(ClassifierModelFactory.MODEL_XGBOOST)
model.fit(X_train, X_valid, y_train, y_valid)
predict = model.predict(X_valid)
print(predict)

predict = [lambda p:1 if p > 0.6 else 0 for p in predict]
print(y_valid)
print(predict)

