# coding: utf-8

import numpy as np


def mse(preds, train_data):
    labels = train_data.get_label().squeeze()
    return 'mse', np.mean((preds.round(2) - labels)**2)/2, False
