import random
import math
import pandas as pd

def bootstrap(x_data, y_data):
    data = pd.concat([x_data, y_data], axis = 1)
    n, m = data.shape
    feature_names = random.sample(data.columns.values[:-1], int(math.sqrt(m - 1)))
    feature_names.append(data.columns.values[-1])
    rows = [random.randint(0, n-1) for _ in range(n)]
    traindata = data.iloc[rows][feature_names]
    return train_data[:-1], train_data[-1]   
