from model_selection.model.light_gbm import LightGbmC
from model_selection.model.xgboost import XgbC


class ClassifierModelFactory(object):

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

    model_name = {MODEL_LIGHET_GBM: 'light_gbm_c_',
                  MODEL_XGBOOST: 'xgboost_c_',
                  MODEL_CAT_BOOST: 'cat_boost_c_',
                  MODEL_RANDOM_FOREST: 'random_forest_c_',
                  MODEL_GBR: 'gb_c_',
                  MODEL_TENSOR_DNN: 'tf_dnn_c_',
                  MODEL_TENSOR_LR: 'tf_lr_c_',
                  MODEL_KNR: 'knn_c_',
                  MODEL_EXTRA_TREE: 'extra_tree_c_',
                  MODEL_DECISION_TREE: 'decision_tree_c_',
                  MODEL_LINEAR: 'linear_c',
                  MODEL_SVM_LR: 'svm_lr_c_',
                  MODEL_SVM_POLY: 'svm_poly_c_',
                  MODEL_SVM_RBF: 'svm_brf_c_',
                  MODEL_SGD: 'sgd_c_'}

    def create_model(self, argument):
        method_name = 'model_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def model_0(self):
        return LightGbmC()

    def model_1(self):
        return XgbC()

    def get_model_name(self, argument):
        return self.model_name[argument]

