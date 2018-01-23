from catboost import CatBoostRegressor

from model_selection.predict_model import PredictModel


class CatBoostR(PredictModel):

    cbr = None

    def create_predict_model(self):
        self.cbr = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3, loss_function='RMSE',
                                     eval_metric='RMSE', random_seed=33)

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        self.cbr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.cbr.predict(X_test)


class CatBoostC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass
