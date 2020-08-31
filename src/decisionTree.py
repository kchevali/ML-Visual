from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import helper as hp
import pandas as pd
from table import Table
from graphics import *
# from random import uniform


class DecisionTree:
    def __init__(self, table):
        self.current = Node(table=table, tree=self)
        self.model = DecisionTreeClassifier()
        self.model.fit(self.getTable().encodedData, self.getTable().encodeTargetCol)

    def getTable(self):
        return self.current.table

    def getClassCount(self):
        return self.current.table.classCount

    def getClass(self, index):
        return self.current.table.classes[index]

    def getColName(self, index):
        return self.current.table.colNames[index]

    def getChildren(self):
        return self.current.children

    def getParent(self):
        return self.current.parent

    def getParentColumn(self):
        return self.current.parent.column

    def getValue(self):
        return self.current.value

    def isParentColumn(self, column):
        if self.current == None or self.current.parent == None:
            return False
        return self.current.parent.containsColumn(column)

    def isCurrentColumn(self, column):
        return self.current and self.current.column == column

    def isVertical(self):
        return type(self.getView().keyDown("div")) != HStack

    def add(self, column):
        self.current.add(column)

    def remove(self):
        self.current.remove()

    def goBack(self):
        self.current = self.current.parent

    def go(self, index):
        self.current = self.current.children[index]

    def predict(self, row):
        return self.current.predict(row)

    def modelPredict(self, row):
        return self.model.predict([row])[0]

    def test(self, testData):
        self.modelTest(testData)
        correct = 0
        for index, row in testData.iterrows():
            if self.predict(row) == row[self.current.table.targetName]:
                correct += 1
        return correct / testData.dataRows

    def modelTest(self, testData):
        y_pred = self.model.predict(testData.encodedData)
        y_test = testData.targetCol
        return metrics.accuracy_score(y_test, y_pred)

    def isRoot(self):
        return self.current and not self.current.parent

    def hasChildren(self):
        return self.current and len(self.current.children) > 0

    def getChild(self, index):
        return self.current.children[index]


class Node:

    def __init__(self, table, tree, parent=None, value=None):
        self.column = None
        self.children = []
        self.table = table
        self.tree = tree
        self.parent = parent
        self.value = value

        # print("CREATE NODE")
        # print(self.dataFrame.head())

    def add(self, column):
        self.column = column
        self.children = []
        for item in self.table[column].unique():
            self.children.append(Node(
                table=Table(data=self.table[self.table[self.column] == item], param=self.table.param),
                tree=self.tree, parent=self, value=item
            ))

    def remove(self):
        self.column = None
        self.children = []

    def containsColumn(self, column):
        if self.column == column:
            return True
        if self.parent != None:
            return self.parent.containsColumn(column)
        return False

    def predict(self, row):
        if self.children:
            for child in self.children:
                # print("Row:", row)
                # print("C:", child.column)
                if child.value == row[self.column]:
                    return child.predict(row)
            return None
        return self.table.commonTarget()


if __name__ == '__main__':
    hp.clear()
    print("Running Decision Tree MAIN")

    fileName = "examples/movie"
    # pd.set_option('display.max_rows', None)

    def createView(sender, index):
        return Label(str(index))

    table = Table(filePath=fileName, createView=createView)
    print("Table\n", table.data)

    # ==============================================
    # Data Frame Usage
    # ==============================================

    # print(end="\nIterate Columns: ")
    # for column in table.colNames:
    #     print(column, end=" ")

    # print(end="\nIterate Items in Column: ")
    # for item in table[table.targetName]:
    #     print(item, end=" ")

    # print(end="\nIterate Rows: ")
    # for index, row in table.iterrows():
    #     print(row[table.targetName], end=" | ")

    # print("\nFirst Item:", table['type'][1])
    # print("Column Count:", table.cols, "Row Count:", table.dataRows)
    # print(end="Unique Values[{}]: ".format(table['type'].nunique()))
    # for item in table['type'].unique():
    #     print(item, end=" ")
    # print(end="\nDirectors of Long Comedies: ")
    # for index, row in table.loc[(table['type'] == 'comedy') & (table['length'] == 'long')].iterrows():
    #     print(row['director'], end=" ")

    featureCols = ['type_animated', 'director_adam', 'director_lass', 'director_singer', 'fam_actors']
    print("\nFeatures:", featureCols)

    x = table.encodedData[featureCols]
    # print("X:\n", x)

    y = table.targetCol
    # print("Y:\n", y)

    # y = table.data['Live Zoo']
    # print("Y:", y, "Shape:", y.shape)

    # x = table.data.loc[:, featureCols]
    # print("Shape:", x.shape)
    # X = [[0, 0], [1, 1]]
    # Y = [0, 1]
    # model = DecisionTree(x, y)
    # tree = model.tree
    tree = DecisionTreeClassifier()
    tree.fit(x, y)

    value = [[True, False, False, True, True]]
    result = tree.predict(value)
    print("Prediction of: {} is {}".format(value, result))

    # n_nodes = tree.tree_.node_count
    # children_left = tree.tree_.children_left
    # children_right = tree.tree_.children_right
    # feature = tree.tree_.feature
    # threshold = tree.tree_.threshold
    # value = tree.tree_.value
    # classes = tree.classes_

    # # print(y[np.where(value[2].flatten() > 0.5)[0]].tolist())

    # # exit()

    # # The tree structure can be traversed to compute various properties such
    # # as the depth of each node and whether or not it is a leaf.
    # node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    # is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    # stack = [(0, -1)]  # seed is the root node id and its parent depth
    # while stack:  # len > 0
    #     node_id, parent_depth = stack.pop()
    #     node_depth[node_id] = parent_depth + 1

    #     # If we have a test node
    #     if (children_left[node_id] != children_right[node_id]):
    #         stack.append((children_left[node_id], parent_depth + 1))
    #         stack.append((children_right[node_id], parent_depth + 1))
    #     else:
    #         is_leaves[node_id] = True

    # print("The binary tree structure has %s nodes and has the following tree structure:" % n_nodes)
    # for i in range(n_nodes):
    #     if is_leaves[i]:
    #         print("%snode=%s leaf node with value %s" % (node_depth[i] * "\t", i, str(classes[np.where(value[i].flatten() > 0.5)[0]].tolist())))
    #     else:
    #         print("%snode=%s test node: go to node %s if %s <= %s else to node %s." % (node_depth[i] * "\t", i,
    #                                                                                    children_left[i], str(featureCols[feature[i]]), threshold[i], children_right[i],))
    # print()

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
