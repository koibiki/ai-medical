def create_sample(data):
    columns = data.columns
    length = data.shape[0]
    count = 0
    for item in data:
        data