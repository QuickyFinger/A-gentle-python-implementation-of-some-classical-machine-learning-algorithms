from ../Decision_Tree/decision_tree_C45 import Decision_Tree_C45
from bootstrap import bootstrap

class random_forest(object):
    def __init__(self, num_trees=None):
        if not isinstance(num_trees.dtype, np.int):
            raise ValueError("number of decision trees should be integer !")
        else:
            self.num_trees = num_trees
    
    def fit(self, x_data, y_data):
        self.forest = dict()
        for index in range(self.num_trees):
            sample_data_x, sample_data_y = bootstrap(x_data, y_data)
            model = Decision_Tree_C45(min_info_gain = 0, x_data = sample_data_x, y_data = sample_data_y)
            tree = model.generate_tree()
            self.forest[index] = tree
    
    def predict(self, test_data):
        # add some code here
    
