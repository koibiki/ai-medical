import xgboost as xgb
from model_selection.predict_model import PredictModel

class_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,  # 如同学习率
    'seed': 1000,
    'nthread': 7,  # cpu 线程数
    'eval_metric': 'auc'
}

regress_params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,  # 如同学习率
    'seed': 1000,
    'nthread': 7,  # cpu 线程数
    'eval_metric': 'rmse'
}


class XgbR(PredictModel):

    xgbr = None

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(xgb_train, 'train'), (xgb_valid, 'val')]
        self.xgbr = xgb.train(regress_params, xgb_train, num_boost_round=5000, evals=watchlist, early_stopping_rounds=100,
                              verbose_eval=100)

    def predict(self, X_test):
        xgb_test = xgb.DMatrix(X_test)
        return self.xgbr.predict(xgb_test, ntree_limit=self.xgbr.best_ntree_limit)


class XgbC(PredictModel):

    xgbc = None

    def create_predict_model(self):
        pass

    def fit(self, X_train, X_valid, y_train, y_valid):
        self.create_predict_model()
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(xgb_train, 'train'), (xgb_valid, 'val')]
        self.xgbc = xgb.train(class_params, xgb_train, num_boost_round=5000, evals=watchlist, early_stopping_rounds=100,
                              verbose_eval=100)
        pass

    def predict(self, X_test):
        xgb_test = xgb.DMatrix(X_test)
        return self.xgbc.predict(xgb_test, ntree_limit=self.xgbc.best_ntree_limit)
