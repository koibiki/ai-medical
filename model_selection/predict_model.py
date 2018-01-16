from abc import abstractmethod


class PredictModel:

    @abstractmethod
    def create_predict_model(self):
        pass

    @abstractmethod
    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    @abstractmethod
    def predict(self, test_X):
        pass
