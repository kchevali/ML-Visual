from sklearn.tree import DecisionTreeClassifier
import numpy as np
import helper as hp
import pandas as pd
from table import Table
from graphics import *
# from random import uniform
from math import sin, cos, pi


class DecisionTree:
    def __init__(self, table):
        self.current = BinaryNode(table=table)
        self.classColors = {}
        for i in range(self.current.classCount):
            self.classColors[self.current.classes[i]] = Color.calmColor(i / self.current.classCount)
        self.current.colorTable(self.classColors)
        self.branch = Branch(view=self.current.getDotViews(self.classColors), disjoint=Container())

        # self.tree = DecisionTreeClassifier()
        # self.tree.fit(x, y)

    # def predict(self, x):
    #     return self.tree.predict(x)

    def getView(self):
        return self.branch.view

    def getContainer(self):
        return self.branch.getContainer()

    def getData(self):
        return self.current.dataFrame

    def getDisjoint(self):
        return self.branch.disjoint

    def getTable(self):
        return self.current.table

    def isParentColumn(self, column):
        if not self.current or not self.current.parent:
            return False
        return self.current.parent.containsColumn(column)

    def isCurrentColumn(self, column):
        return self.current and self.current.value == column

    def isVertical(self):
        return type(self.getView().keyDown("div")) != HStack

    def add(self, column):
        self.current.add(column)
        container = self.getContainer()
        prevStack = type(self.getDisjoint().keyUp("div"))
        stack = VStack if prevStack != VStack else HStack
        label = [Label(text=self.current.parent.value, fontSize=20, color=Color.green if self.current.boolean else Color.red, dx=-0.95, dy=-1)] if self.current.parent else []
        self.branch.setView(view=ZStack(views=label + [
                            stack(views=[
                                self.current.left.getDotViews(self.classColors),
                                self.current.right.getDotViews(self.classColors),
                            ], ratios=[
                                (self.current.left.classCount + 1) / (self.current.classCount + 2),
                                (self.current.right.classCount + 1) / (self.current.classCount + 2)
                            ], border=20 if self.current.parent else 0,
                                keywords="div")
                            ], keywords=["z", "left" if self.current.boolean else "right"])
                            )

    def remove(self):
        self.current.remove()
        self.branch.setView(view=self.current.getDotViews(self.classColors))

    def goBack(self):
        self.current = self.current.parent
        self.branch.move(self.getDisjoint().keyUp("z"))

    def goLeft(self):
        self.current = self.current.left
        self.branch.move(self.getView().keyDown("left", excludeSelf=True))

    def goRight(self):
        self.current = self.current.right
        print("\nView", self.getView(), self.getView().keywords)
        print("Bottom:", self.getView().keyDown("right", excludeSelf=True))
        self.branch.move(self.getView().keyDown("right", excludeSelf=True))

    def isRoot(self):
        return self.current and self.current.parent == None

    def hasLeft(self):
        return self.current and self.current.left != None

    def hasRight(self):
        return self.current and self.current.right != None


class BinaryNode:

    def __init__(self, parent=None, table=None, boolean=None, classColors={}):
        self.remove()
        self.parent = parent
        self.table = table
        self.boolean = boolean
        self.dataFrame = table.data if table else None
        self.classCount = self.dataFrame['label'].nunique()
        self.classes = self.dataFrame.label.unique()
        self.colorTable(classColors)

        # print("CREATE NODE")
        # print(self.dataFrame.head())

    def colorTable(self, classColors):
        self.classColors = classColors
        if self.classColors:  # if len > 0
            for i in range(self.table.rows - 1):
                rect = self.table.getView((i + 1) * self.table.cols).keyDown("rect")
                rect.color = classColors[self.dataFrame['label'][i]]
                rect.isHidden = False

    def add(self, column):
        self.value = column
        self.left = BinaryNode(parent=self, table=Table(self.dataFrame[self.dataFrame[self.value] == True], fontSize=20), boolean=True, classColors=self.classColors)
        self.right = BinaryNode(parent=self, table=Table(self.dataFrame[self.dataFrame[self.value] == False], fontSize=20), boolean=False, classColors=self.classColors)

    def remove(self):
        self.value = None
        self.left = None  # Yes
        self.right = None  # No

    def getDotViews(self, colors):
        if self.classCount:
            trig = 2.0 * pi / self.classCount
        label = [Label(text=self.parent.value, fontSize=20, color=Color.green if self.boolean else Color.red, dx=-0.95, dy=-1)] if self.parent else []
        return ZStack(views=[
            Ellipse(color=colors[self.classes[i]],
                    strokeColor=Color.red, strokeWidth=2,
                    dx=0.5 * cos(trig * i) if self.classCount > 1 else 0.0,
                    dy=0.5 * sin(trig * i) if self.classCount > 1 else 0.0,
                    border=0,
                    lockedWidth=20, lockedHeight=20
                    ) for i in range(self.classCount)
        ] + label, keywords="left" if self.boolean else "right")

    def containsColumn(self, column):
        if self.value == column:
            return True
        if self.parent:
            return self.parent.containsColumn(column)
        return False


if __name__ == '__main__':
    hp.clear()
    print("Running Decision Tree MAIN")

    fileName = "examples/movie"
    # pd.set_option('display.max_rows', None)
    table = Table(filePath=fileName, fontSize=20)
    print("Table\n", table.data)

    print(end="\nIterate Columns: ")
    for column in table.colNames:
        print(column, end=" ")

    print(end="\nIterate Items in Column: ")
    for item in table[table.targetCol]:
        print(item, end=" ")

    print(end="\nIterate Rows: ")
    for index, row in table.iterrows():
        print(row[table.targetCol], row[0], end=" | ")

    print("\nFirst Item:", table['type'][1])
    print("Column Count:", table.cols, "Row Count:", table.dataRows)
    print(end="Unique Values[{}]: ".format(table['type'].nunique()))
    for item in table['type'].unique():
        print(item, end=" ")
    print(end="\nDirectors of Long Comedies: ")
    for index, row in table.loc[(table['type'] == 'comedy') & (table['length'] == 'long')].iterrows():
        print(row['director'], end=" ")

    # featureCols = ['is_bug', 'can_fly', 'live_farm']
    # print("\nFeatures:", featureCols)

    # x = table.data[featureCols]
    # print("X:\n", x)

    # y = table.data.label
    # print("Y:\n", y)

    # # y = table.data['Live Zoo']
    # # print("Y:", y, "Shape:", y.shape)

    # # x = table.data.loc[:, featureCols]
    # # print("Shape:", x.shape)
    # # X = [[0, 0], [1, 1]]
    # # Y = [0, 1]
    # # model = DecisionTree(x, y)
    # # tree = model.tree
    # tree = DecisionTreeClassifier()
    # tree.fit(x, y)

    # value = [[False, False, True]]
    # result = tree.predict(value)
    # print("Prediction of: {} is {}".format(value, result))

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
