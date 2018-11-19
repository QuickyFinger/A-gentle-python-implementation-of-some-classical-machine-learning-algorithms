from GBDT import GBDT
import pandas as pd

def demo():
    dataset = pd.read_csv('../training_data.csv')
    test_data = pd.read_csv('../test_data.csv')
    model = GBDT()
    model.fit(dataset)
    result = model.predict_value(test_data)

if __name__ == '__main__':
    demo()
