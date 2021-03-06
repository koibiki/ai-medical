from sklearn.ensemble import ExtraTreesRegressor

from model_selection.predict_model import PredictModel


class ExtraTreeR(PredictModel):

    etr = None

    def create_predict_model(self):
        self.etr = ExtraTreesRegressor()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        self.etr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.etr.predict(X_test)


class ExtraTreeC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
