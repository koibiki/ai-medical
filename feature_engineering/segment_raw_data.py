import pandas as pd

segment_types = {
    1: 1,
    2: 2,
    3: 2,
    4: 2,
}

segment_params = {
    1: [12, 20, 45, 65],
    2: 38,
    3: 55,
}


def segment_3(x, params, prefix):
    if x <= params[0]:
        return prefix + '0s'
    elif x > params[0] and x < params[1]:
        return prefix + '1s'
    else:
        return prefix + '2s'


def segment_5(x, params, prefix):
    if x <= params[0]:
        return prefix + '0s'
    elif x > params[0] and x <= params[1]:
        return prefix + '1s'
    elif x > params[1] and x <= params[2]:
        return prefix + '2s'
    elif x > params[2] and x <= params[3]:
        return prefix + '3s'
    else:
        return prefix + '4s'


def segment_raw_data(data, index):
    segment_param = segment_params[index]
    segment_type = segment_types[index]
    if segment_type == 1:
        data['segment_' + str(index)] = data.apply(lambda x: segment_5(x, segment_param, 'segment_' + str(index)))
        return pd.get_dummies(data['segment_' + str(index)])
    elif segment_type == 2:
        data['segment_' + str(index)] = data.apply(lambda x: 1 if x >= segment_param else 0)
        return data['segment_' + str(index)]
    elif segment_type == 3:
        data['segment_' + str(index)] = data.apply(lambda x: segment_3(x, segment_param, 'segment_' + str(index)))
        return pd.get_dummies(data['segment_' + str(index)])
    else:
        raise Exception("Invalid level!", 1)

