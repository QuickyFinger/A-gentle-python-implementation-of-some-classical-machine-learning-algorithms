# gradient boosting decision tree
# 使用 CART回归树 作为基树

from CART import CART
from predict import predict

class GBDT(object):
    
    def _init_(self, max_depth=5, learning_rate=1e-1, max_iter=5):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.model = {}
    
    def compute_gradient(self, y_truth, y_predict):
        # using least square error
        grad = y_truth - y_predict
        return grad
    
    def fit(self, dataset):
        data = deepcopy(dataset)
        y_truth = data[data.columns[-1]]
        x_data = data[data.columns[0:-1]]
        # step 1, compute f_0
        base_model = CART(max_depth=5)
        self.model[0] = base_model.generate_tree(x_data, y_truth)
        y_predict = predict(x_data, self.model[0])
        for i in range(1, self.max_iter):
            #base_model = CART(max_depth=5)
            residual = self.compute_gradient(y_truth, y_predict)
            self.model[i] = base_model.generate_tree(x_data, residual)
            y_predict += self.learning_rate * predict(x_data, self.model[i])
        return self.model
    
    def predict_value(self, x_data):
        y_predict = predict(x_data, self.model[0])
        for i in range(1, self.max_iter):
            y_predict += self.learning_rate * predict(x_data, self.model[i])
        return y_predict