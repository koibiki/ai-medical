from model_selection.model.cat_boost import CatBoostR
from model_selection.model.decision_tree import DecisionTreeR
from model_selection.model.extra_tree import ExtraTreeR
from model_selection.model.gradient_boost import GbR
from model_selection.model.knn import KnnR
from model_selection.model.light_gbm import LightGbmR
from model_selection.model.linear import LinearR
from model_selection.model.random_forest import RandomForestR
from model_selection.model.sgdr import SgdR
from model_selection.model.skflow_dnn import SkflowDnnR
from model_selection.model.skflow_lr import SkflowLrR
from model_selection.model.svm_lr import SvrLrR
from model_selection.model.svm_poly import SvrPolyR
from model_selection.model.svm_rbf import SvrRbfR
from model_selection.model.xgboost import XgbR


class RegressorModelFactory(object):

    MODEL_LIGHET_GBM = 0
    MODEL_XGBOOST = 1
    MODEL_CAT_BOOST = 2
    MODEL_RANDOM_FOREST = 3
    MODEL_GBR = 4
    MODEL_TENSOR_DNN = 5
    MODEL_TENSOR_LR = 6
    MODEL_KNR = 7
    MODEL_EXTRA_TREE = 8
    MODEL_SGD = 9
    MODEL_LINEAR = 10
    MODEL_SVM_LR = 11
    MODEL_SVM_POLY = 12
    MODEL_SVM_RBF = 13
    MODEL_DECISION_TREE = 14

    model_name = {MODEL_LIGHET_GBM: 'light_gbm_r_',
                  MODEL_XGBOOST: 'xgboost_r_',
                  MODEL_CAT_BOOST: 'cat_boost_r_',
                  MODEL_RANDOM_FOREST: 'random_forest_r_',
                  MODEL_GBR: 'gb_r_',
                  MODEL_TENSOR_DNN: 'tf_dnn_r_',
                  MODEL_TENSOR_LR: 'tf_lr_r_',
                  MODEL_KNR: 'knn_r_',
                  MODEL_EXTRA_TREE: 'extra_tree_r_',
                  MODEL_DECISION_TREE: 'decision_tree_r_',
                  MODEL_LINEAR: 'linear_r',
                  MODEL_SVM_LR: 'svm_lr_r_',
                  MODEL_SVM_POLY: 'svm_poly_r_',
                  MODEL_SVM_RBF: 'svm_brf_r_',
                  MODEL_SGD: 'sgd_r_'}

    def create_model(self, argument):
        method_name = 'model_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def model_0(self):
        return LightGbmR()

    def model_1(self):
        return XgbR()

    def model_2(self):
        return CatBoostR()

    def model_3(self):
        return RandomForestR()

    def model_4(self):
        return GbR()

    def model_5(self):
        return SkflowDnnR()

    def model_6(self):
        return SkflowLrR()

    def model_7(self):
        return KnnR()

    def model_8(self):
        return ExtraTreeR()

    def model_9(self):
        return SgdR()

    def model_10(self):
        return LinearR()

    def model_11(self):
        return SvrLrR()

    def model_12(self):
        return SvrPolyR()

    def model_13(self):
        return SvrRbfR()

    def model_14(self):
        return DecisionTreeR()

    def get_model_name(self, argument):
        return self.model_name[argument]


