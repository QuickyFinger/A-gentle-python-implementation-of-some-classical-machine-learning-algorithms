from decision_tree_CART import CART
from decision_tree_cut import CART_CUT
from decision_tree_predict import *
from decision_tree_data import toy_data
from decision_tree_plot import PlotTree

def demo():
    data = toy_data()
    x_data = data.drop(data.columns[-1], axis=1)
    y_data = data[data.columns[-1]]
    decision_tree = CART(0, x_data, y_data)
    tree_node = decision_tree.generate_tree()

    tree_cut = CART_CUT()
    child_tree_list = tree_cut.tree_cut(tree_node)
    test_data = deepcopy(x_data)
    best_acc = 0
    for tree in child_tree_list:
        CART_tree_predict(test_data, tree)
        result = test_data['supervised'] == y_data
        acc = result.sum() / len(result)
        if acc >= best_acc:
            best_acc = acc
            best_tree = tree
    print('best acc: ', best_acc)

    pt = PlotTree()
    pt.createPlot(tree_node)

if __name__ == '__main__':
    demo()
