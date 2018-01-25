from model_selection.model.light_gbm import LightGbmMultiC
from model_selection.model.xgboost import XgbMultiC


class MultiClassifierModelFactory(object):

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

    model_name = {MODEL_LIGHET_GBM: 'light_gbm_mc_',
                  MODEL_XGBOOST: 'xgboost_mc_',
                  MODEL_CAT_BOOST: 'cat_boost_mc_',
                  MODEL_RANDOM_FOREST: 'random_forest_mc_',
                  MODEL_GBR: 'gb_mc_',
                  MODEL_TENSOR_DNN: 'tf_dnn_mc_',
                  MODEL_TENSOR_LR: 'tf_lr_mc_',
                  MODEL_KNR: 'knn_mc_',
                  MODEL_EXTRA_TREE: 'extra_tree_mc_',
                  MODEL_DECISION_TREE: 'decision_tree_mc_',
                  MODEL_LINEAR: 'linear_mc',
                  MODEL_SVM_LR: 'svm_lr_mc_',
                  MODEL_SVM_POLY: 'svm_poly_mc_',
                  MODEL_SVM_RBF: 'svm_brf_mc_',
                  MODEL_SGD: 'sgd_mc_'}

    def create_model(self, argument):
        method_name = 'model_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def model_0(self):
        return LightGbmMultiC()

    def model_1(self):
        return XgbMultiC()

    def get_model_name(self, argument):
        return self.model_name[argument]

