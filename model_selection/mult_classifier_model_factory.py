from model_selection.model.light_gbm import LightGbmMultiC
from model_selection.model.xgboost import XgbMultiC


class MultiClassifierModelFactory(object):

    MODEL_LIGHET_GBM = 0
    MODEL_XGBOOST = 1
    # MODEL_GBR = 2
    # MODEL_SVM = 3
    # MODEL_TENSOR_DNN = 4
    # MODEL_TENSOR_LR = 5
    # MODEL_KNR = 6
    # MODEL_LINEAR = 7

    def create_model(self, argument):
        method_name = 'model_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def model_0(self):
        return LightGbmMultiC()

    def model_1(self):
        return XgbMultiC()

    def get_model_name(self, argument):
        method_name = 'get_model_name_' + str(argument)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    def get_model_name_0(self):
        return 'light_gbm_multi_c_'

    def get_model_name_1(self):
        return 'xgboost_multi_c_'
