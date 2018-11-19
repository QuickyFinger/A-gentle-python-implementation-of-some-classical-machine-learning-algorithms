import pandas as pd
import numpy as np
from TreeNode import Node

class Decision_Tree_ID3(object):
    def __init__(self, min_info_gain, x_data, y_data):
        self.min_info_gain = min_info_gain
        self.root_node = Node(x_data=x_data, y_data=y_data)
        self.feature_item = self.get_feature(x_data)
    
    def get_feature(self, x_data):
        feature_item = {}
        for feature in list(x_data.columns):
            feature_item[feature] = list(x_data[feature].value_counts().keys())
        return feature_item

    
    def label_gain(self, y_data):
        tmp = y_data.value_counts(normalize=True)
        label_entr = tmp.apply(lambda x: -1 * x * np.log2(x)).sum()
        return label_entr
        
    def info_gain(self, label_entr, feature_name, label_name, data):
        tmp1 = data.groupby([feature_name, label_name], as_index=False)[label_name].agg({'count':'count'})
        tmp2 = tmp1.groupby([feature_name], as_index=False)['count'].agg({'sum':'sum'})
        tmp3 = pd.merge(tmp1, tmp2, on=feature_name, how='left')
        tmp3['prob'] = tmp3.apply(lambda x: x['count'] / x['sum'], axis=1)
        total = tmp3['sum'].sum()
        tmp3['entr'] = tmp3.apply(lambda x: -1 * x['count'] * np.log2(x['prob']), axis=1)
        cond_entr = tmp3['entr'].sum() / total
        info_gain = (label_entr - cond_entr) / label_entr
        return info_gain
    
    def _Decision_Tree(self, tree_node):
        x_data = tree_node.x_data
        y_data = tree_node.y_data
        
        feature_names = list(x_data.columns)
        xy_data = pd.concat([x_data, y_data], axis=1)
        label_name =  list(xy_data.columns)[-1]
        label_unique = xy_data[label_name].unique()
        
        if len(label_unique) == 1:
            cate = y_data.loc[0]
            tree_node.category = cate
            return tree_node
        
        if len(feature_names) == 0:
            cate = y_data.value_counts(ascending=False).keys()[0]
            tree_node.category = cate
            return tree_node
        
        label_entr = self.label_entr(y_data)
        
        max_gain = 0
        for feature in feature_names:
            info_gain = self.info_gain(label_entr, feature, label_name, xy_data)
            if info_gain > max_gain:
                max_gain = info_gain
                f_name = feature
                
        if max_gain <= self.min_info_gain:
            cate = y_data.value_counts(ascending=False).keys()[0]
            tree_node.category = cate
            return tree_node
        
        tree_node.feature = f_name
        tree_node.children = dict()
        
        for sub_attribute in self.feature_item[tree_node.feature]:
            sub_data = xy_data[xy_data[f_name] == sub_attribute]
            sub_data_x = sub_data.drop(list(sub_data.columns)[-1], axis=1)
            sub_data_x.drop(tree_node.feature, axis=1, inplace=True)
            sub_data_y = sub_data[list(sub_data.columns)[-1]]
            child_node = Node(tree_node, None, None, None, sub_data_x, sub_data_y)
            tree_node.children[sub_attribute] = Decision_Tree(child_node)
        return tree_node
    
    def generate_tree(self):
        tree = self._Decision_Tree(self.tree_node)
return tree
