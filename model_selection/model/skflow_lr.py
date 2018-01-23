import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import LinearRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from model_selection.predict_model import PredictModel


class SkflowLrR(PredictModel):

    ss = None
    dnn = None
    feature_columns = None

    def input_fn(self, X_train, y_train):
        feature_cols = {k: tf.constant(X_train[k].values)for k in self.feature_columns}
        labels = tf.constant(y_train.values)
        return feature_cols, labels

    def create_predict_model(self):
        self.ss = MaxAbsScaler()
        print()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()

        self.feature_columns = X_train.columns
        tf_feature_cols = [tf.contrib.layers.real_valued_column(k) for k in self.feature_columns]

        ss_X_train = self.ss.fit_transform(X_train)
        ss_X_train = pd.DataFrame(ss_X_train, columns=self.feature_columns)

        self.dnn = LinearRegressor(feature_columns=tf_feature_cols)
        self.dnn.fit(input_fn=lambda: self.input_fn(ss_X_train, y_train), steps=1600)

    def predict(self, X_test):
        X_test = self.ss.transform(X_test)
        X_test_df = pd.DataFrame(X_test, columns=self.feature_columns)
        predict = self.dnn.predict(input_fn=lambda: self.input_fn(X_test_df, pd.DataFrame(np.zeros(len(X_test)))),
                                   as_iterable=False)
        return predict


class SkflowLrC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
