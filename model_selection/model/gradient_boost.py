from sklearn.ensemble import GradientBoostingRegressor

from model_selection.predict_model import PredictModel


class GbR(PredictModel):

    rfr = None

    def create_predict_model(self):
        self.rfr = GradientBoostingRegressor()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        self.rfr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.rfr.predict(X_test)


class GbC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
