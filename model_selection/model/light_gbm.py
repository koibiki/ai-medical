import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from model_selection.predict_model import PredictModel

regress_params = {
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

class_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


multi_class_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


class LightGbmR(PredictModel):

    @staticmethod
    def evaluator(pred, df):
        label = df.get_label().values.copy()
        score = mean_squared_error(label, pred) * 0.5
        return '0.5mse', score, False

    gbm = None

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        self.gbm = lgb.train(regress_params, lgb_train, num_boost_round=50000, valid_sets=lgb_valid, verbose_eval=200,
                             feval=self.evaluator, early_stopping_rounds=300)

    def predict(self, X_test):
        return self.gbm.predict(X_test)

    def feature_importance(self):
        return self.gbm.feature_importance(importance_type='split')


class LightGbmC(PredictModel):

    gbm = None

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        self.gbm = lgb.train(class_params, lgb_train, num_boost_round=50000, valid_sets=lgb_valid, verbose_eval=200,
                             early_stopping_rounds=300)

    def predict(self, X_test):
        return self.gbm.predict(X_test)



class LightGbmMultiC(PredictModel):

    gbm = None

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        self.gbm = lgb.train(multi_class_params, lgb_train, num_boost_round=50000, valid_sets=lgb_valid,
                             verbose_eval=200, early_stopping_rounds=300)

    def predict(self, X_test):
        return self.gbm.predict(X_test)


