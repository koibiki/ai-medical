from sklearn.tree import DecisionTreeRegressor

from model_selection.predict_model import PredictModel


class DecisionTreeR(PredictModel):

    dtr = None

    def create_predict_model(self):
        self.dtr = DecisionTreeRegressor()

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        self.dtr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.dtr.predict(X_test)


class DecisionTreeC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
