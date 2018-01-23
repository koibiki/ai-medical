from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler

from model_selection.predict_model import PredictModel


class LinearR(PredictModel):

    lr = None
    x_mas = None

    def create_predict_model(self):
        self.lr = LinearRegression()
        self.x_mas = MaxAbsScaler()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        X_train = self.x_mas.fit_transform(X_train)
        self.lr.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.x_mas.transform(X_test)
        return self.lr.predict(X_test)


class LinearC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
