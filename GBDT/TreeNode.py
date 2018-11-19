class TreeNode(object):
    def __init__(self, split_feature=None, split_value=None, left_node=None, right_node=None, value=None):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        self.value = value