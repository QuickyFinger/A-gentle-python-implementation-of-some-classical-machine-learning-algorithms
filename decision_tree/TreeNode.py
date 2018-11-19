class Node(object):
    def __init__(self, parent=None, children=None, feature=None, category=None, x_data=None, y_data=None):
        self.parent = parent
        self.children = children
        self.feature = feature
        self.category = category
        self.x_data = x_data
        self.y_data = y_data
    
    def get_category(self):
        return self.category
    
    def other_function(self):
        pass
