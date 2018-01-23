from sklearn.neighbors import KNeighborsRegressor

from model_selection.predict_model import PredictModel


class KnnR(PredictModel):

    knr = None

    def create_predict_model(self):
        self.knr = KNeighborsRegressor(weights='distance')

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        self.knr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.knr.predict(X_test)


class KnnC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
