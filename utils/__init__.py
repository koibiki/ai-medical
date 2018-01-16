import random
import pandas as pd


def create_sample(data):
    randoms = []
    for index in range(len(data)):
        random_value = [item * (0.95 + 0.1 * random.random()) for item in data.iloc[index].values]
        random_series = pd.Series(random_value, index=data.columns)
        randoms.append(random_series)
    return pd.DataFrame(randoms)
