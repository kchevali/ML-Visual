from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, linear_model
from sklearn.metrics import mean_squared_error
from scipy import spatial
from myModels import MyLinear
import numpy as np
import helper as hp
import pandas as pd
from table import Table
from graphics import *
from math import inf, sqrt, log
import math
# from random import uniform


class Model:
    def __init__(self, table, testing=None, partition=0.7, color=Color.red, **kwargs):
        super().__init__(**kwargs)
        self.training, self.testing = table.partition(partition) if testing is None else (table, testing)
        self.color = color
        self.validateData()
        # self.compModel = None -- maybe should make new class - think how to handle

    def predict(self, row):
        pass

    def test(self, testTable=None):
        # use test set
        pass

    def validateData(self):
        # check conditions - number of x, y, labels...etc per class
        pass

    def error(self, testTable=None):
        pass


class Classifier(Model):
    # takes multiple features and outputs a categorical data

    # "Accuracy"
    def test(self, testTable=None):
        if testTable == None:
            testTable = self.testing
        correct = 0
        for row in testTable:
            correct += 1 if self.predict(row) == row[0] else 0
        return (correct / testTable.rows)

    def error(self, testTable=None):
        return 1.0 - self.test(testTable)


class Regression(Model):
    # takes multiple features and outputs real data

    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.reset()

    # "Error"

    def test(self, testTable=None):
        if testTable == None:
            testTable = self.testing
        error = 0
        for _, x, y in testTable:
            error += (y - self.getY(x))**2
        return sqrt(error) / testTable.rows

    def error(self, testTable=None):
        return self.test(testTable)

    def getY(self, x):
        pass

    def getX(self, y):
        pass

    def reset(self):
        self.points = []
        self.cef = [0] * self.length  # highest power first
        self.debug = 0

    def getEdgePoints(self):
        allPts = [
            (self.training.minX, self.getY(self.training.minX)),
            (self.training.maxX, self.getY(self.training.maxX)),
            (self.getX(self.training.minY), self.training.minY),
            (self.getX(self.training.maxY), self.training.maxY),
        ]
        return [(self.training.createDisplayX(x), self.training.createDisplayY(y)) for x, y in allPts if(x >= self.training.minX and x <= self.training.maxX and y >= self.training.minY and y <= self.training.maxY)]

    def getManyPoints(self):
        pts = []
        length, num = 100, -1
        delta = 2 / length
        for i in range(length):
            pts.append((self.training.createDisplayX(num), self.training.createDisplayY(self.getY(num))))
            num += delta
        return pts

    def addPoint(self, point, storePoint=True):
        if storePoint:
            if len(self.points) < self.n + 1:
                self.points.append(point)
            else:
                self.reset()

    def getEqString(self):
        return "EQ"


class DecisionTree(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current = DTNode(table=self.training, tree=self)
        self.training = None

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

    def isLockedColumn(self, column):
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


class DTNode:

    def __init__(self, table, tree, parent=None, value=None):
        self.column = None
        self.children = []
        self.training = table
        self.tree = tree
        self.parent = parent
        self.value = value

        # print("CREATE NODE")
        # print(self.dataFrame.head())

    def add(self, column):
        self.column = column
        self.children = []
        for item in self.training[column].unique():
            self.children.append(DTNode(
                table=Table(data=self.training[self.training[self.column] == item], param=self.training.param),
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
        return self.training.commonTarget()


class KNN(Classifier):

    def __init__(self, k=3, bestK=False, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        if bestK:
            self.findBestK()

        # xy = self.training.getArray([1, 2])
        # print(xy)
        # self.distTree = spatial.KDTree(xy)

    def getNeighbor(self, x, y):
        neighbors = []
        pt = [x, y]
        out = spatial.KDTree(self.training.getArray([1, 2])).query(pt, k=self.k)[1]

        # out = self.distTree.query(pt, k=self.k)[1]
        return out if out is np.ndarray else np.array([out])

    def predict(self, row):
        return self.predictPoint(row[1], row[2])

    def predictPoint(self, X, Y):
        # print("Predicting:", X, Y)
        record = {}
        for rowIndex in self.getNeighbor(X, Y):
            # print("Neighbor:", x2, y2)

            try:
                label, x, y = self.training.data[rowIndex]
                # print("M:", x, y)
                if label in record:
                    record[label] += 1
                else:
                    record[label] = 1
            except:
                pass

        maxClassCount = 0
        bestClass = None
        for className, classCount in record.items():
            if classCount > maxClassCount:
                maxClassCount = classCount
                bestClass = className
        return bestClass

    def findBestK(self):
        self.k = 1
        acc = self.test()
        bestAcc = 0
        while(acc > bestAcc):
            self.k += 2
            bestAcc = acc
            acc = self.test()
        self.k -= 1
        print("Final:", self.test())


class Linear(Regression):
    def __init__(self, n=1, alpha=0.05, epsilon=1e-5, **kwargs):
        super().__init__(length=self.n + 1, **kwargs)
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.llamda = 0.1
        self.dJ = self.epsilon

        self.model = linear_model.LinearRegression()
        self.model.fit([self.x], [self.y])

    # incoming point must be pixel coordinates

    def addPoint(self, point, storePoint=True):
        # point = (hp.map(point[0] - self.offsetX, 0, self.width, -1, 1), hp.map(point[1] - self.offsetY, 0, self.height, -1, 1))
        super.addPoint(point, storePoint)
        if len(self.points) > 0:
            if self.n == 1:
                x1, y1 = self.points[0]
                x2, y2 = point if len(self.points) == 1 else self.points[1]
                if x2 != x1:
                    self.getLinearEq(x1, y1, x2, y2)
                    return self.getEdgePoints()
            if self.n == 2:
                x1, y1 = self.points[0]
                x2, y2 = point if len(self.points) == 1 else self.points[1]
                x3, y3 = (-point[0], point[1]) if len(self.points) == 1 else (point if len(self.points) == 2 else self.points[2])

                if x2 != x1 and x3 != x1 and x3 != x2:
                    self.getQuadEq(x1, y1, x2, y2, x3, y3)
                    return self.getManyPoints()

                    # edgePoints = [(0, self.getY(0)), (width, self.getY(width)), (self.getX(0)[0], 0), (self.getX(height)[0], height)]
                    # # print("Edge:", edgePoints)
                    # return [(x + offsetX, y + offsetY) for x, y in edgePoints if(x != None and x >= 0 and x <= width and y >= 0 and y <= height)]
        return None

    def getX(self, y):
        if self.n == 1:
            slope, intercept = tuple(self.cef)
            return (y - intercept) / slope if slope != 0 else inf
        elif self.n == 2:
            a, b, c = tuple(self.cef)
            delta = b * b - 4 * a * c
            if delta < 0 or a == 0:
                return None, None
            return (-b + sqrt(delta)) / (2 * a), (-b - sqrt(delta)) / (2 * a)

    def getY(self, x):
        y = 0
        x_ = 1
        for i in range(self.n, -1, -1):
            y += self.cef[i] * x_
            x_ *= x
        return y

    def getModelError(self, testTable):
        colA, colB = self.training.columns[1], self.training.columns[2]
        predY = self.model.predict([self.x])
        # print("COF:")
        # print(len(self.model.coef_))
        # print(len(self.model.intercept_))
        error = mean_squared_error([self.x], predY)
        # print("ERR:", error)
        return error

    def getQuadEq(self, x1, y1, x2, y2, x3, y3):
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
        self.cef = [a, b, c]

    def getLinearEq(self, x1, y1, x2, y2):
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        self.cef = [slope, intercept]

    def fit(self):
        if self.dJ >= self.epsilon:
            newCefs = [self.cef[i] - self.alpha * self.getJGradient(degreeX=self.n - i) for i in range(self.n + 1)]
            # a = self.cef[0] - self.alpha * self.getJGradient(multX=True) / self.length
            # b = self.cef[1] - self.alpha * self.getJGradient(multX=False) / self.length
            self.dJ = self.getJ(self.cef) - self.getJ(newCefs)
            self.cef = newCefs
            # print("NUMS:", self.cef, self.dJ)
            return True
        print("FIT DONE")
        return False

    def getJ(self, cef):
        total = 0
        # testTotal = 0
        for i in range(self.length):
            localTotal = self.y[i]
            for j in range(self.n + 1):
                localTotal -= cef[j] * (self.x[i]**(self.n - j))

            # testTotal += (self.y[i] - cef[0] * self.x[i] - cef[1])**2
            total += localTotal * localTotal
        # print("J:", total, testTotal)
        return total

    def getJGradient(self, degreeX):
        total = 0
        # testTotal = 0
        for i in range(self.length):
            localTotal = -self.y[i]
            for j in range(self.n + 1):
                localTotal += self.cef[j] * (self.x[i]**(self.n - j))
            total += localTotal * (self.x[i]**degreeX)
            # testTotal += (self.cef[0] * self.x[i] + self.cef[1] - self.y[i]) * (self.x[i]**degreeX)
        # print("G:", total, testTotal)
        total /= self.length
        # print("Gradient Step:", total)
        return total

    def fitLasso(self):
        if self.dJ >= self.epsilon:
            newCefs = [self.cef[i] - self.alpha * self.getJGradient(degreeX=self.n - i) / self.length for i in range(self.n + 1)]
            # a = self.cef[0] - self.alpha * self.getJGradient(multX=True) / self.length
            # b = self.cef[1] - self.alpha * self.getJGradient(multX=False) / self.length
            self.dJ = self.getJ(self.cef) - self.getJ(newCefs)
            self.cef = newCefs
            # print(self.cef, self.dJ)
            return True
        return False

    def getJLasso(self, cef):
        total = 0
        # testTotal = 0
        for i in range(self.length):
            localTotal = self.y[i]
            for j in range(self.n + 1):
                localTotal -= cef[j] * (self.x[i]**(self.n - j))

            # testTotal += (self.y[i] - cef[0] * self.x[i] - cef[1])**2
            total += localTotal * localTotal
        for j in range(self.n + 1):
            total += abs(cef[j]) * self.llamda
        # print("J:", total, testTotal)
        return total

    def getJGradientLasso(self, degreeX):
        total = 0
        # testTotal = 0
        for i in range(self.length):
            localTotal = -self.y[i]
            for j in range(self.n + 1):
                localTotal += self.cef[j] * (self.x[i]**(self.n - j))
            total += localTotal * (self.x[i]**degreeX)
            # testTotal += (self.cef[0] * self.x[i] + self.cef[1] - self.y[i]) * (self.x[i]**degreeX)
        for j in range(self.n + 1):
            total -= self.llamda * self.cef[j] / abs(self.cef[j])
        # print("G:", total, testTotal)
        return total

    def getEqString(self):
        out = "Y="
        n = self.n
        for i in range(self.n + 1):
            val = round(self.cef[i], 2)
            out += ("" if val < 0 or i == 0 else "+") + \
                (str(val) if val != 0 or n == 0 else "") + \
                ("x" if n > 0 else "") + \
                (("^" + str(n)) if n > 1 else "")
            n -= 1
        return out


class Logistic(Regression):

    def __init__(self, **kwargs):
        super().__init__(length=2, **kwargs)
        # self.compModel = linear_model.LogisticRegression()

    def addPoint(self, point, storePoint=True):
        # point = (hp.map(point[0] - self.offsetX, 0, self.width, 0, 1), hp.map(point[1] - self.offsetY, 0, self.height, 0, 1))
        super.addPoint(point, storePoint)

        if len(self.points) > 0:
            x1, y1 = self.points[0]
            x2, y2 = point if len(self.points) == 1 else self.points[1]
            if x2 != x1:
                self.getSigmoidEq(x1, y1, x2, y2)
                # pts = []
                # # print("")
                # length, num = 100, 0
                # delta = 1.0 / length
                # for i in range(length):
                #     pts.append((hp.map(num, 0, 1, 0, self.width) + self.offsetX, hp.map(self.getY(num), 0, 1, 0, self.height) + self.offsetY))
                #     num += delta
                #     # print(i + self.offsetX, self.getY(i) + self.offsetY)
                # return pts
                return self.getManyPoints()

        return None

    # y = height / (1 + e^(ax+b))

    def invY(self, y):
        return log((1.0 - y) / y)  # * self.height

    def getX(self, y):
        return (self.invY(y) - self.cef[1]) / self.cef[0]

    def getY(self, x):
        try:
            exp = math.e**(x * self.cef[0] + self.cef[1])
            return (1.0 / (1.0 + exp))
        except:
            return 1.0

    def getSigmoidEq(self, x1, y1, x2, y2):
        slope = (self.invY(y2) - self.invY(y1)) / (x2 - x1)
        intercept = self.invY(y1) - slope * x1
        self.cef = [slope, intercept]
        # print("CEF:", self.cef)

    def getEqString(self):
        return "LOG EQ"


if __name__ == '__main__':
    hp.clear()
    print("Running Models MAIN")
    table = Table(filePath="examples/decisionTree/small.csv")
    knn = KNN(table=table, partition=0.8, k=1)
    print(knn.test())


#     fileName = "examples/movie"
#     # pd.set_option('display.max_rows', None)

#     def createView(sender, index):
#         return Label(str(index))

#     table = Table(filePath=fileName, createView=createView)
#     print("Table\n", table.data)

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

# featureCols = ['type_animated', 'director_adam', 'director_lass', 'director_singer', 'fam_actors']
# print("\nFeatures:", featureCols)

# x = table.encodedData[featureCols]
# print("X:\n", x)

# y = table.targetCol
# print("Y:\n", y)

# y = table.data['Live Zoo']
# print("Y:", y, "Shape:", y.shape)

# x = table.data.loc[:, featureCols]
# print("Shape:", x.shape)
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# model = DecisionTree(x, y)
# tree = model.tree
# tree = DecisionTreeClassifier()
# tree.fit(x, y)

# value = [[True, False, False, True, True]]
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
