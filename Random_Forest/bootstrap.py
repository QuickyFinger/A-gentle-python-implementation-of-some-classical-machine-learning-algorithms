import random
import math

def bootstrap(data):
    n, m = data.shape
    feature_names = random.sample(data.columns.values[:-1], int(math.sqrt(m - 1)))
    feature_names.append(data.columns.values[-1])
    rows = [random.randint(0, n-1) for _ in range(n)]
    traindata = data.iloc[rows][feature_names]
    return train_data, feature_names
    
