from sklearn.tree import DecisionTreeClassifier
import numpy as np
import helper as hp
from table import Table


class DecisionTree:
    def __init__(self, x, y):
        self.tree = DecisionTreeClassifier()
        self.tree.fit(x, y)

    def predict(self, x):
        return self.tree.predict(x)


def getNames(arr, labels):
    return labels[np.nonzero(arr[i] > 0.5)]


if __name__ == '__main__':
    hp.clear()
    print("Running Decision Tree MAIN")

    table = Table.readCSV("examples/animal.csv")
    print("Table\n", table.data.head())

    featureCols = ['is_bug', 'can_fly', 'live_farm']
    print("\nFeatures:", featureCols)

    x = table.data[featureCols]
    print("X:\n", x)

    y = table.data.label
    print("Y:\n", y)

    # y = table.data['Live Zoo']
    # print("Y:", y, "Shape:", y.shape)

    # x = table.data.loc[:, featureCols]
    # print("Shape:", x.shape)
    # X = [[0, 0], [1, 1]]
    # Y = [0, 1]
    model = DecisionTree(x, y)
    tree = model.tree

    value = [[False, False, True]]
    result = tree.predict(value)
    print("Prediction of: {} is {}".format(value, result))

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    classes = tree.classes_

    # print(y[np.where(value[2].flatten() > 0.5)[0]].tolist())

    # exit()

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has the following tree structure:" % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node with value %s" % (node_depth[i] * "\t", i, str(classes[np.where(value[i].flatten() > 0.5)[0]].tolist())))
        else:
            print("%snode=%s test node: go to node %s if %s <= %s else to node %s." % (node_depth[i] * "\t", i,
                                                                                       children_left[i], str(featureCols[feature[i]]), threshold[i], children_right[i],))
    print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    # node_indicator = tree.decision_path(x)

    # # Similarly, we can also have the leaves ids reached by each sample.

    # leave_id = tree.apply(x)

    # # Now, it's possible to get the tests that were used to predict a sample or
    # # a group of samples. First, let's make it for the sample.

    # sample_id = 0
    # node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
    #                                     node_indicator.indptr[sample_id + 1]]

    # print('Rules used to predict sample %s: ' % sample_id)
    # for node_id in node_index:
    #     if leave_id[sample_id] == node_id:
    #         continue

    #     if (x[sample_id, feature[node_id]] <= threshold[node_id]):
    #         threshold_sign = "<="
    #     else:
    #         threshold_sign = ">"

    #     print("decision id node %s : (x[%s, %s] (= %s) %s %s)"
    #           % (node_id,
    #              sample_id,
    #              feature[node_id],
    #              x[sample_id, feature[node_id]],
    #              threshold_sign,
    #              threshold[node_id]))

    # # For a group of samples, we have the following common node.
    # sample_ids = [0, 1]
    # common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
    #                 len(sample_ids))

    # common_node_id = np.arange(n_nodes)[common_nodes]

    # print("\nThe following samples %s share the node %s in the tree"
    #       % (sample_ids, common_node_id))
    # print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
