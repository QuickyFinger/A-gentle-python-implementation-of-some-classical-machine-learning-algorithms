import pandas as pd
from TreeNode import Node

class CART(object):
    def __init__(self, stop_criter, x_data, y_data):
        self.stop_criter = stop_criter
        self.root_node = Node(x_data=x_data, y_data=y_data)
        self.feature_item = self.get_feature(x_data)
        
    def get_feature(self, x_data):
        feature_item = {}
        for feature in list(x_data.columns):
            feature_item[feature] = list(x_data[feature].value_counts().keys())
        return feature_item
    
    def compute_gini(self, data):
        y_data = data[data.columns[-1]].value_counts(normalize=True)
        y_data = y_data.apply(lambda x: x * x)
        gini = 1 - y_data.sum()
        return gini
    
    def _generate_tree(self, tree_node):
        x_data = tree_node.x_data
        y_data = tree_node.y_data
        
        min_gini = float('inf')
        features = list(x_data.columns)
        xy_data = pd.concat([x_data, y_data], axis=1)
        label_gini = self.compute_gini(xy_data)
        
        if label_gini <= self.stop_criter:
            cate = y_data.value_counts(ascending=False).keys()[0]
            tree_node.category = cate
            return tree_node
        
        if len(features) == 0:
            cate = y_data.value_counts(ascending=False).keys()[0]
            tree_node.category = cate
            return tree_node
        
        if len(list(y_data.unique())) == 0:
            cate = y_data.iloc[0]
            tree_node.category = cate
            return tree_node
            
        best_feature = None
        
        for feature in features:
            total_sample = len(x_data)
            for item in self.feature_item[feature]:
                sub_data_yes = xy_data[xy_data[feature] == item]
                sub_data_no = xy_data[xy_data[feature] != item]
                
                item_sample_yes = len(sub_data_yes)
                item_sample_no  = len(sub_data_no)
                
                gini_yes = self.compute_gini(sub_data_yes)
                gini_no = self.compute_gini(sub_data_no)
                
                gini = (item_sample_yes / total_sample) * gini_yes + (item_sample_no / total_sample) * gini_no
                
                if gini < min_gini:
                    min_gini = gini
                    best_feature = feature
                    best_item = item
                    
        if best_feature == None:
            cate = y_data.value_counts(ascending=False).keys()[0]
            tree_node.category = cate
            return tree_node
        
        tree_node.feature = best_feature
        # left node
        X_data = xy_data[xy_data[best_feature] == best_item]
        Y_data = X_data[X_data.columns[-1]]
        X_data.drop(X_data.columns[-1], axis=1, inplace=True)
        X_data.drop(best_feature, axis=1, inplace=True)
        child_node = Node(tree_node, None, None, None, X_data, Y_data)
        tree_node.children = {}
        tree_node.children[str(best_item)] = self._generate_tree(child_node)
        
        # right node
        X_data = xy_data[xy_data[best_feature] != best_item]
        Y_data = X_data[X_data.columns[-1]]
        X_data.drop(X_data.columns[-1], axis=1, inplace=True)
        child_node = TreeNode(tree_node, None, None, None, X_data, Y_data)
        tree_node.children["!" + str(best_item)] = self._generate_tree(child_node)
        
        return tree_node
        
    def generate_tree(self):
        return self._generate_tree(self.root_node)
