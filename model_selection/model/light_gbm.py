import lightgbm as lgb
from model_selection.predict_model import PredictModel

class_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


class LightGbmR(PredictModel):

    gbm = None

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        self.gbm = lgb.train(class_params, lgb_train, num_boost_round=20000, valid_sets=lgb_valid, verbose_eval=100,
                             early_stopping_rounds=100)

    def predict(self, X_test):
        return self.gbm.predict(X_test)


class LightGbmC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass


class LightGbmMultiC(PredictModel):

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        pass

    def predict(self, X_test):
        pass

