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
    def __init__(self, table, color=Color.red, isLinear=False, isCategorical=False, isConnected=False, displayRaw=False, **kwargs):
        self.xs = table.xs
        self.y = table.y
        self.minX, self.maxX = table.minX(), table.maxX()
        self.minY, self.maxY = table.minX(1), table.maxX(1)
        self.color = color
        self.isLinear = isLinear
        self.isCategorical = isCategorical
        self.isConnected = isConnected
        self.displayRaw = displayRaw
        self.classColors = table.classColors


class Classifier(Model):
    # takes multiple features and outputs a categorical data
    def __init__(self, isCategorical=True, **kwargs):
        super().__init__(isLinear=False, isConnected=False, isCategorical=isCategorical, **kwargs)

    def accuracy(self, testTable):
        correct = 0
        xs, y = testTable.xs, testTable.y
        for i in range(testTable.rowCount):
            correct += 1 if self.predict(xs[i]) == y[i] else 0
        return (correct / testTable.rowCount)

    def predict(self, _xs):
        pass


class Regression(Model):
    # takes multiple features and outputs real data

    def __init__(self, length, isCategorical=False, **kwargs):
        super().__init__(isConnected=True, isCategorical=isCategorical, **kwargs)
        self.length = length
        self.reset()

    def error(self, testTable):
        error = 0
        x, y = testTable.getXY()
        for i in range(testTable.rowCount):
            error += (y[i] - self.getY(x[i]))**2
        return sqrt(error) / testTable.rowCount

    def getY(self, x):
        pass

    def getX(self, y):
        pass

    def reset(self):
        self.points = []
        self.cef = [0] * self.length  # highest power first
        self.debug = 0

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

    def __init__(self, table, k=3, bestK=False, **kwargs):
        super().__init__(table=table, displayRaw=True, **kwargs)
        self.k = k
        self.kdTree = spatial.KDTree(np.array(table.xData))
        if bestK:
            self.findBestK()

        # xy = self.training.getArray([1, 2])
        # print(xy)
        # self.distTree = spatial.KDTree(xy)

    def getNeighbor(self, _xs):
        return np.reshape(self.kdTree.query(_xs, k=self.k)[1], self.k)
        # return out if out is np.ndarray else np.array([out])

    def predict(self, _xs):
        ys = np.array([self.y[i] for i in self.getNeighbor(_xs)])
        return np.argmax(np.bincount(ys))

    def findBestK(self, testTable):
        acc = self.accuracy(testTable=testTable)
        bestAcc = 0
        while(acc > bestAcc):
            self.k += 2
            bestAcc = acc
            acc = self.accuracy(testTable=testTable)
            if(acc <= bestAcc):
                self.k -= 2


class Linear(Regression):
    def __init__(self, n=1, alpha=0.05, epsilon=1e-5, **kwargs):
        super().__init__(length=self.n + 1, isLinear=n == 1, **kwargs)
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.llamda = 0.1
        self.dJ = self.epsilon

    # incoming point must be pixel coordinates

    def getEq(self, points):
        if len(points) > 0:
            x1, y1 = points[0]
            if len(points) > 1:
                x2, y2 = points[1]
                if self.n == 1 and x2 != x1:
                    self.getLinearEq(x1, y1, x2, y2)
                if len(points) > 2 and self.n == 2:
                    x3, y3 = points[2]
                    if x2 != x1 and x3 != x1 and x3 != x2:
                        self.getQuadEq(x1, y1, x2, y2, x3, y3)

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
                localTotal -= cef[j] * (self.xs[i]**(self.n - j))

            # testTotal += (self.y[i] - cef[0] * self.xs[i] - cef[1])**2
            total += localTotal * localTotal
        # print("J:", total, testTotal)
        return total

    def getJGradient(self, degreeX):
        total = 0
        # testTotal = 0
        for i in range(self.length):
            localTotal = -self.y[i]
            for j in range(self.n + 1):
                localTotal += self.cef[j] * (self.xs[i]**(self.n - j))
            total += localTotal * (self.xs[i]**degreeX)
            # testTotal += (self.cef[0] * self.xs[i] + self.cef[1] - self.y[i]) * (self.xs[i]**degreeX)
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
                localTotal -= cef[j] * (self.xs[i]**(self.n - j))

            # testTotal += (self.y[i] - cef[0] * self.xs[i] - cef[1])**2
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
                localTotal += self.cef[j] * (self.xs[i]**(self.n - j))
            total += localTotal * (self.xs[i]**degreeX)
            # testTotal += (self.cef[0] * self.xs[i] + self.cef[1] - self.y[i]) * (self.xs[i]**degreeX)
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
        super().__init__(length=2, isLinear=False, **kwargs)
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


class SVM(Regression):
    def __init__(self, C=0.01, n_iters=100, learning_rate=0.1, **kwargs):
        super().__init__(length=0, isLinear=False, **kwargs)
        self.c = C
        self.iter = n_iters
        self.eta = learning_rate

        w = np.zeros([1, self.xs.shape[1]])
        b = 0

        costs = np.zeros(self.iter)
        for i in range(self.iter):
            cost = self.xs @ w.T + b
            b = b - self.eta * self.c * sum(cost - self.y)
            w = w - self.eta * self.c * sum((cost - self.y) * self.xs)
            costs[i] = self.c * sum((self.y * cost) + (1 - self.y) * cost) + (1 / 2) * sum(w.T**2)

        self.w = w
        self.b = b
        self.costs = costs

    def predict(self, x_test):
        pred_y = []
        svm = x_test @ self.w.T + self.b
        for i in svm:
            if i >= 0:
                pred_y.append(1)
            else:
                pred_y.append(0)

        return pred_y

    def getY(self, _xs, line=0):
        y = (line - self.b - _xs * self.w.T[0]) / self.w.T[1]
        return y[0]

    def getX(self, _xs, line=0):
        return (line - self.b - _xs * self.w.T[1]) / self.w.T[0]


if __name__ == '__main__':
    hp.clear()
    print("Running Models MAIN")
    table = Table(filePath="examples/decisionTree/small").createXXYTable()
    train, test = table.partition()
    knn = KNN(table=train, partition=0.8, k=3)
    print("Accuracy:", 100 * knn.accuracy(testTable=test))
    for _xs in knn.xs:
        print(_xs, end=" Closest: ")
        for j in knn.getNeighbor(_xs):
            print(knn.xs[j], knn.y[j], end=" | ")
        print()
