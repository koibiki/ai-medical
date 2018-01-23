from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MaxAbsScaler

from model_selection.predict_model import PredictModel


class SgdR(PredictModel):

    sgdr = None
    mas = None

    def create_predict_model(self):
        self.sgdr = SGDRegressor()
        self.mas = MaxAbsScaler()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        X_train = self.mas.fit_transform(X_train)
        self.sgdr.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.mas.transform(X_test)
        return self.sgdr.predict(X_test)


class LinearC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
