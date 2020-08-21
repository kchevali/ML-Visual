from sklearn.tree import DecisionTreeClassifier
import numpy as np
import helper as hp
import pandas as pd
from table import Table
from graphics import *
# from random import uniform
from math import sin, cos, pi


class DecisionTree:
    def __init__(self, table, backMethod, goMethod):
        self.backMethod = backMethod
        self.goMethod = goMethod
        self.classColors = {}
        for i in range(table.classCount):
            self.classColors[table.classes[i]] = Color.calmColor(i / table.classCount)

        self.current = Node(table=table, tree=self)
        self.branch = Branch(view=self.current.getDotViews(), disjoint=Container())

        # self.tree = DecisionTreeClassifier()
        # self.tree.fit(x, y)

    # def predict(self, x):
    #     return self.tree.predict(x)

    def getView(self):
        return self.branch.view

    def getContainer(self):
        return self.branch.getContainer()

    def getDisjoint(self):
        return self.branch.disjoint

    def getTable(self):
        return self.current.table

    def getClassCount(self):
        return self.current.table.classCount

    def getClass(self, index):
        return self.current.table.classes[index]

    def getColName(self, index):
        return self.current.table.colNames[index]

    def isParentColumn(self, column):
        if not self.current or not self.current.parent:
            return False
        return self.current.parent.containsColumn(column)

    def isCurrentColumn(self, column):
        return self.current and self.current.column == column

    def isVertical(self):
        return type(self.getView().keyDown("div")) != HStack

    def add(self, column, backMethod, goMethod):
        self.current.add(column)
        container = self.getContainer()
        prevStack = type(self.getDisjoint().keyUp("div"))
        stack = VStack if prevStack != VStack else HStack
        label = [Button(view=Label(text="{}:{}".format(self.current.parent.column, self.current.value), fontSize=20,
                                   color=Color.white, dx=-0.95, dy=-1), run=self.backMethod)] if self.current.parent else []
        totalClassCount = len(self.current.children)
        for child in self.current.children:
            totalClassCount += child.table.classCount

        self.branch.setView(view=ZStack(views=label + [
            stack(views=[
                Button(view=self.current.children[i].getDotViews(), run=self.goMethod, tag=i) for i in range(len(self.current.children))
            ], ratios=[
                (child.table.classCount + 1) / totalClassCount for child in self.current.children
            ], border=20 if self.current.parent else 0, keywords="div")
        ], keywords=["z"]))

    def remove(self):
        self.current.remove()
        self.branch.setView(view=self.current.getDotViews())

    def goBack(self):
        self.current = self.current.parent
        self.branch.move(self.getDisjoint().keyUp("z"))

    def go(self, index):
        self.current = self.current.children[index]
        container = self.getView().keyDown("div")[index]
        stack = container.keyDown("z")
        self.branch.move(stack if stack else container.keyDown("dotStack"))

    def isRoot(self):
        return self.current and not self.current.parent

    def hasChildren(self):
        return self.current and len(self.current.children) > 0


class Node:

    def __init__(self, table, tree, parent=None, value=None):
        self.column = None
        self.children = []
        self.table = table
        self.tree = tree
        self.parent = parent
        self.value = value

        index = 1
        for item in self.table.targetCol:
            print("Index:", index * self.table.cols)
            rect = self.table.getView(index * self.table.cols).keyDown("rect")
            rect.color = self.tree.classColors[item]
            rect.isHidden = False
            index += 1

        # print("CREATE NODE")
        # print(self.dataFrame.head())
    def add(self, column):
        self.column = column
        self.children = []
        for item in self.table[column].unique():
            self.children.append(Node(
                table=Table(self.table[self.table[self.column] == item], param=self.table.param, fontSize=20),
                tree=self.tree, parent=self, value=item
            ))

    def remove(self):
        self.column = None
        self.children = []

    def getDotViews(self):
        if self.table.classCount:
            trig = 2.0 * pi / self.table.classCount
        label = [Label(text="{}:{}".format(self.parent.column, self.value), fontSize=20, color=Color.white, dx=-0.95, dy=-1)] if self.parent else []
        return ZStack(views=[
            Ellipse(color=self.tree.classColors[self.table.classes[i]],
                    strokeColor=Color.red, strokeWidth=2,
                    dx=0.5 * cos(trig * i) if self.table.classCount > 1 else 0.0,
                    dy=0.5 * sin(trig * i) if self.table.classCount > 1 else 0.0,
                    border=0,
                    lockedWidth=20, lockedHeight=20
                    ) for i in range(self.table.classCount)
        ] + label, keywords="dotStack")

    def containsColumn(self, column):
        if self.column == column:
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
        print(row[table.targetName], row[0], end=" | ")

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
