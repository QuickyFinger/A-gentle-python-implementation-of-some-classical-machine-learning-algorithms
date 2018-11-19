# CART回归树
from TreeNode import TreeNode

class CART(object):
    def _init_(self, max_depth=5, min_mse=0.):
        self.max_depth = max_depth
        self.min_mse = min_mse
    
    def MSE(data):
        if len(data) < 2:
            return 0
        y_data = data[data.columns[-1]]
        mean = y_data.mean()
        error = y_data.apply(lambda x: (x - mean) * (x - mean)).sum()
        return error
    
    def _generate_tree(self, x_data, y_data, depth, tree_node):
        
        if depth < self.max_depth:
            xy_data = pd.concat([x_data, y_data], axis=1)
            label_mse = self.MSE(y_data)
            features = list(x_data.columns)

            if label_mse < self.min_mse:
                mean = y_data.mean()
                tree_node.value = mean
                return tree_node

            if len(features) == 0:
                mean = y_data.mean()
                tree_node.value = mean
                return tree_node

            if len(list(y_data.unique())) == 0:
                value = y_data.iloc[0]
                tree_node.value = value
                return tree_node

            best_feature = None
            _mse = float('inf')

            for feature in features:
                for item in list(x_data[feature].value_counts().keys()):
                    left_data = xy_data[xy_data[feature] <= item]
                    right_data = xy_data[xy_data[feature] > item]
                    sum_mse = self.MSE(left_data) + self.MSE(right_data)
                    if sum_mse < _mse:
                        _mse = sum_mse
                        best_feature = feature
                        best_item = item


            if best_feature == None:
                mean = y_data.mean()
                tree_node.value = mean
                return tree_node

            tree_node.split_feature = best_feature
            tree_node.split_value = best_item
            left_data = xy_data[xy_data[best_feature] <= best_item]
            right_data = xy_data[xy_data[best_feature] > best_item]
            left_tree_node = TreeNode()
            right_tree_node = TreeNode()
            tree_node.left_node = self._generate_tree(left_data[left_data.columns[0:-1]], left_data[left_data.columns[-1]], depth+1, left_tree_node)
            tree_node.right_node = self._generate_tree(right_data[right_data.columns[0:-1]], right_data[right_data.columns[-1]], depth+1, right_tree_node)
            return tree_node
        
        else:
            mean = y_data.mean()
            tree_node.value = mean
            return tree_node
        
    def generate_tree(self, x_data, y_data):
        tree_node = TreeNode()
        return self._generate_tree(x_data, y_data, 0, tree_node)
        
    def find_categoryCART(self, row, tree_node):
        childNode = tree_node.children
        if tree_node.feature == None:
            row['supervised'] = tree_node.category
            return
        else:
            node = childNode.get(row[tree_node.feature])
            if node == None:
                node = childNode.get(list(childNode.keys())[1])
            find_categoryCART(row, node)
            return

    def predict(self, test_data, tree_node):
        num = len(test_data)
        test_data['supervised'] = None
        for i in range(num):
            row = test_data.loc[i]
            find_categoryCART(row, tree_node)
            test_data.loc[i]['supervised'] = row['supervised']