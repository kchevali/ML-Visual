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
from math import inf, sqrt, log, pi, sin, cos
import math
# from random import uniform


class Model:
    def __init__(self, table, testingTable=None, color=Color.red, drawTable=False, isUserSet=False, isClassification=False, isRegression=False, **kwargs):
        self.table = table
        self.testingTable = testingTable
        self.color = color
        self.drawTable = drawTable
        # isLinear=False, isCategorical=False, isConnected=False, displayRaw=False
        # self.isLinear = isLinear
        # self.isCategorical = isCategorical
        # self.isConnected = isConnected
        # self.displayRaw = displayRaw
        self.minX1, self.maxX1 = None, None
        self.minX2, self.maxX2 = None, None
        self.colNameA, self.colNameB = None, None
        self.graphics = []  # stores Point objects
        self.graphicsDict = {}
        self.isUserSet = isUserSet
        self.isRunning = False
        self.isClassification = isClassification
        self.isRegression = isRegression

    def getTablePtsXX(self):
        return [self.getPt(self.table.x[i][0], self.table.x[i][1], self.table.classColors[self.table.y[i]]) for i in range(self.table.rowCount)]

    def getTablePtsXY(self):
        return [self.getPt(self.table.x[i][0], self.table.y[i], self.color) for i in range(self.table.rowCount)]

    def getPt(self, x, y, color):  # get many points
        return (hp.map(x, self.minX1, self.maxX1, -1, 1), hp.map(y, self.minX2, self.maxX2, -1, 1), color)

    def addGraphics(self, *args):
        for graphic in args:
            if type(graphic) == tuple:
                key, graphic = graphic
                self.graphicsDict[key] = graphic
            self.graphics.append(graphic)

    def getGraphic(self, key):
        return self.graphicsDict[key] if key in self.graphicsDict else None

    def startTraining(self):
        self.reset()
        self.isRunning = True

    def getScoreString(self):
        raise NotImplementedError("Please Implement getScoreString")

    def defaultScoreString(self):
        raise NotImplementedError("Please Implement defaultScoreString")

    def reset(self):
        raise NotImplementedError("Please Implement reset")


class Classifier(Model):
    # takes multiple features and outputs a categorical data
    def __init__(self, **kwargs):
        super().__init__(isLinear=False, isConnected=False,isClassification=True, **kwargs)
        self.minX1, self.maxX1 = self.table.minX(), self.table.maxX()
        self.minX2, self.maxX2 = self.table.minX(self.table.xNames[1]), self.table.maxX(self.table.xNames[1])
        self.colNameA, self.colNameB = self.table.xNames[0], self.table.xNames[1]

    def accuracy(self, testTable):
        correct = 0
        x, y = testTable.x, testTable.y
        for i in range(testTable.rowCount):
            correct += 1 if self.predict(x[i]) == y[i] else 0
        return (correct / testTable.rowCount)

    def predict(self, _x):
        pass

    def getScoreString(self):
        return str(round(100 * self.accuracy(testTable=self.testingTable), 2)) + "%"

    def defaultScoreString(self):
        return "Accuracy: --"

    def getCircleLabelPts(self, table=None):  # circle
        if table == None:
            table = self.table

        rowCount = table.rowCount
        if rowCount > 0:
            trig = 2.0 * pi / rowCount

        pts = []
        i = 0
        for label in table.y:
            pts.append((0.5 * cos(trig * i) if rowCount > 1 else 0.0,
                        0.5 * sin(trig * i) if rowCount > 1 else 0.0,
                        table.classColors[label]))
            i += 1

        # if treeNode.parent != None:
        #     items.append(Label(text="{}:{}".format(treeNode.parent.column, treeNode.value), fontSize=20, color=Color.white, dx=-0.95, dy=-1))
        # return ZStack(items=items, keywords="dotStack", limit=150)
        return pts


class Regression(Model):
    # takes multiple features and outputs real data

    def __init__(self, length, **kwargs):
        super().__init__(isConnected=True,isRegression=True, **kwargs)
        self.length = length
        self.minX1, self.maxX1 = self.table.minX(), self.table.maxX()
        self.minX2, self.maxX2 = self.table.minY(), self.table.maxY()
        self.colNameA, self.colNameB = self.table.xNames[0], self.table.yName
        if self.drawTable:
            self.graphics.append(Points(pts=self.getTablePtsXY(), color=self.color, isConnected=False))
        self.reset()

    def error(self, testTable):
        error = 0
        for i in range(testTable.rowCount):
            error += (testTable.y[i] - self.getY(testTable.x[i]))**2
        return sqrt(error) / testTable.rowCount

    def reset(self):
        self.critPts = []
        self.cef = [0] * self.length  # highest power first

        pts = self.getGraphic("pts")
        if pts != None:
            pts.reset()

    def getY(self, x):
        raise NotImplementedError("Please Implement getY")

    def getX(self, y):
        raise NotImplementedError("Please Implement getX")

    def getEqString(self):
        raise NotImplementedError("Please Implement getEqString")

    def getEq(self):
        raise NotImplementedError("Please Implement getEq")

    def getScoreString(self):
        return str(round(self.error(testTable=self.testingTable), 4))

    def defaultScoreString(self):
        return "Error: --"

    def cefString(self, constant, power, showPlus=True, roundValue=2):
        constant = round(constant, roundValue)
        if constant == 0:
            return ""
        return ("+" if constant > 0 and showPlus else "") + str(constant) + ("" if power <= 0 else ("x" + ("" if power == 1 else hp.superscript(power))))

    def getPts(self, start=None, end=None, count=40):  # get many points
        if start == None:
            start = self.minX1
        if end == None:
            end = self.maxX1
        return [self.getPt(num, self.getY(num), self.color) for num in hp.rangx(start, end, (end - start) / count, outputEnd=True)]

    def addPt(self, x, y, storePt=True):
        if len(self.critPts) == 0 and not storePt:
            return
        if len(self.critPts) < self.length:
            self.critPts.append((x, y))
            self.getEq()

            # update graphics
            self.getGraphic("pts").setPts(self.getPts())
            self.getGraphic("eq").setFont(text=self.getEqString())
            self.getGraphic("err").setFont(text="Error: " + self.getScoreString())
            if not storePt:
                self.critPts.pop()
        elif storePt:
            self.reset()


class DecisionTree(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.curr = DTNode(table=self.table)
        self.graphics.append(Points(pts=self.getCircleLabelPts(), color=self.color, isConnected=False))
        # self = DecisionTreeClassifier()
        # self.fit(self.getTable().encodedData, self.getTable().encodeTargetCol)

    def getTable():
        return self.curr.table

    def getChildren(self):
        return self.curr.children

    def getParent(self):
        return self.curr.parent

    def getParentColumn(self):
        return self.curr.parent.colIndex

    def getValue(self):
        return self.curr.value

    def isVertical(self):
        return type(self.getView().keyDown("div")) != HStack

    def add(self, colIndex):
        self.curr.setCol(colIndex)

    def remove(self):
        self.curr.reset()

    def goBack(self):
        self.curr = self.curr.parent

    def go(self, index):
        print("Index:", index, "Children Length Before Go:", len(self.curr.children))
        self.curr = self.curr.children[index]

    def predict(self, row):
        return self.curr.predict(row)

    def modelPredict(self, row):
        return self.predict([row])[0]

    def modelTest(self, testData):
        y_pred = self.predict(testData.encodedData)
        y_test = testData.targetCol
        return metrics.accuracy_score(y_test, y_pred)

    def isRoot(self):
        return self.curr and not self.curr.parent

    def hasChildren(self):
        return self.curr and len(self.curr.children) > 0

    def getChild(self, index):
        return self.curr.children[index]


class DTNode:

    def __init__(self, table, parent=None, value=None):
        self.table = table
        self.parent = parent
        self.value = value
        self.colIndex = None
        self.children = []
        # self.tree = tree

    def setCol(self, colIndex):
        self.colIndex = colIndex
        self.children = []
        for item in self.table.uniqueVals(self.colIndex):
            self.children.append(DTNode(
                table=self.table.matchValue(colIndex=self.colIndex, value=item),
                parent=self, value=item
            ))
        print("Children Length after Split:", len(self.children))

    def reset(self):
        self.colIndex = None
        self.children = []

    def predict(self, x):
        if self.children:
            for child in self.children:
                if child.value == x[self.colIndex]:
                    return child.predict(x)
            return None
        return self.table.majorityInTargetColumn()


class KNN(Classifier):

    def __init__(self, table, k=3, bestK=False, **kwargs):
        super().__init__(table=table, displayRaw=True, **kwargs)
        self.k = k
        self.kdTree = spatial.KDTree(np.array(table.x))
        if self.drawTable:
            self.graphics.append(Points(pts=self.getTablePtsXX(), color=self.color, isConnected=False))
        if bestK:
            self.findBestK()

        # xy = self.table.getArray([1, 2])
        # print(xy)
        # self.distTree = spatial.KDTree(xy)

    def getNeighbor(self, _x):
        return np.reshape(self.kdTree.query(_x, k=self.k)[1], self.k)  # if type(_x) == np.ndarray else np.array(_x)

    def predict(self, _x):
        ys = np.array([self.table.y[i] for i in self.getNeighbor(_x)])
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
    def __init__(self, n=1, alpha=1e-3, epsilon=1e-3, **kwargs):
        super().__init__(length=n + 1, isLinear=n == 1, **kwargs)
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.llamda = 0.1
        self.dJ = self.epsilon

    # incoming point must be pixel coordinates

    def getEq(self):
        if len(self.critPts) > 0:
            x1, y1 = self.critPts[0]
            if len(self.critPts) > 1:
                x2, y2 = self.critPts[1]
                if self.n == 1 and x2 != x1:
                    self.getLinearEq(x1, y1, x2, y2)
                if len(self.critPts) > 2 and self.n == 2:
                    x3, y3 = self.critPts[2]
                    if x2 != x1 and x3 != x1 and x3 != x2:
                        self.getQuadEq(x1, y1, x2, y2, x3, y3)

    def getX(self, y):
        if self.n == 1:
            slope, intercept = tuple(self.cef)
            return (y - intercept) / slope if slope != 0 else inf
        elif self.n == 2:
            a, b, c = tuple(self.cef)
            delta = b * b - 4 * a * (c - y)
            # print("Y:", y, "Delta:", delta)
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
            self.getGraphic("pts").setPts(self.getPts())
            self.getGraphic("eq").setFont(text=self.getEqString())
            self.getGraphic("err").setFont(text="Error: " + self.getScoreString())
            return
        print("FIT DONE")
        self.isRunning = False

    def getJ(self, cef):
        total = 0
        # testTotal = 0
        for i in range(self.table.rowCount):
            localTotal = self.table.y[i]
            for j in range(self.n + 1):
                localTotal -= cef[j] * (self.table.x[i][0]**(self.n - j))

            # testTotal += (self.table.y[i] - cef[0] * self.table.x[i] - cef[1])**2
            total += localTotal * localTotal
        # print("J:", total, testTotal)
        return total

    def getJGradient(self, degreeX):
        total = 0
        # testTotal = 0
        for i in range(self.table.rowCount):
            localTotal = -self.table.y[i]
            for j in range(self.n + 1):
                localTotal += self.cef[j] * (self.table.x[i][0]**(self.n - j))
            total += localTotal * (self.table.x[i][0]**degreeX)
            # testTotal += (self.cef[0] * self.table.x[i] + self.cef[1] - self.table.y[i]) * (self.table.x[i]**degreeX)
        # print("G:", total, testTotal)
        total /= self.table.rowCount
        # print("Gradient Step:", total)
        return total

    def fitLasso(self):
        if self.dJ >= self.epsilon:
            newCefs = [self.cef[i] - self.alpha * self.getJGradient(degreeX=self.n - i) / self.table.rowCount for i in range(self.n + 1)]
            # a = self.cef[0] - self.alpha * self.getJGradient(multX=True) / self.table.rowCount
            # b = self.cef[1] - self.alpha * self.getJGradient(multX=False) / self.table.rowCount
            self.dJ = self.getJ(self.cef) - self.getJ(newCefs)
            self.cef = newCefs
            # print(self.cef, self.dJ)
            return True
        return False

    def getJLasso(self, cef):
        total = 0
        # testTotal = 0
        for i in range(self.table.rowCount):
            localTotal = self.table.y[i]
            for j in range(self.n + 1):
                localTotal -= cef[j] * (self.table.x[i]**(self.n - j))

            # testTotal += (self.table.y[i] - cef[0] * self.table.x[i] - cef[1])**2
            total += localTotal * localTotal
        for j in range(self.n + 1):
            total += abs(cef[j]) * self.llamda
        # print("J:", total, testTotal)
        return total

    def getJGradientLasso(self, degreeX):
        total = 0
        # testTotal = 0
        for i in range(self.table.rowCount):
            localTotal = -self.table.y[i]
            for j in range(self.n + 1):
                localTotal += self.cef[j] * (self.table.x[i]**(self.n - j))
            total += localTotal * (self.table.x[i]**degreeX)
            # testTotal += (self.cef[0] * self.table.x[i] + self.cef[1] - self.table.y[i]) * (self.table.x[i]**degreeX)
        for j in range(self.n + 1):
            total -= self.llamda * self.cef[j] / abs(self.cef[j])
        # print("G:", total, testTotal)
        return total

    def getEqString(self):
        out = "Y="
        n = self.n
        for i in range(self.n + 1):
            out += self.cefString(constant=self.cef[i], power=n, showPlus=i > 0)
            n -= 1
        return out

    def getPts(self):
        edgePts = [
            (self.minX1, self.getY(self.minX1)),
            (self.maxX1, self.getY(self.maxX1)),
            (self.getX(self.minX2), self.minX2),
            (self.getX(self.maxX2), self.maxX2),
        ]
        if self.n == 1:
            return [self.getPt(x, y, self.color) for x, y in edgePts if(x >= self.minX1 and x <= self.maxX1 and y >= self.minX2 and y <= self.maxX2)]
        # print("Edge:", edgePts)
        edges = []
        for x, y in edgePts:
            if type(x) != tuple:
                x = [x]
            if type(y) != tuple:
                y = [y]
            edges += [(x_, y_) for x_ in x for y_ in y if x_ != None and y_ != None and x_ >= self.minX1 and x_ <= self.maxX1 and y_ >= self.minX2 and y_ <= self.maxX2]
        edgePts = edges
        edgePts.sort()
        # print("X1:", self.minX1, self.maxX1)
        # print("X2:", self.minX2, self.maxX2)
        # print("Edge:", edgePts)
        # if(self.n > 1):
        #     edgePts = [

        #     ]
        # print("COEFFs:", self.cef)
        # print("Processed:", edgePts)
        # print()
        out = []
        for i in range(1, len(edgePts), 2):
            # sweep
            out += super().getPts(start=edgePts[i - 1][0], end=edgePts[i][0], count=20)
        return out
        # return [self.getPt(x, y, self.color) for x, y in edgePts]


class Logistic(Regression):

    def __init__(self, **kwargs):
        super().__init__(length=2, isLinear=False, **kwargs)
        # self.compModel = linear_model.LogisticRegression()

    def getEq(self):
        if len(self.critPts) > 1:
            x1, y1 = self.critPts[0]
            x2, y2 = self.critPts[1]
            if x2 != x1:
                self.getSigmoidEq(x1, y1, x2, y2)

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
        val1 = self.cef[1]
        return "1/(1+e^(" + self.cefString(self.cef[0], 1, showPlus=False) + self.cefString(self.cef[1], 0) + "))"


class SVM(Regression):
    def __init__(self, C=0.01, n_iters=100, learning_rate=0.1, **kwargs):
        super().__init__(length=0, isLinear=False, **kwargs)
        self.c = C
        self.iter = n_iters
        self.eta = learning_rate

        w = np.zeros([1, self.table.x.shape[1]])
        b = 0

        costs = np.zeros(self.iter)
        for i in range(self.iter):
            cost = self.table.x @ w.T + b
            b = b - self.eta * self.c * sum(cost - self.table.y)
            w = w - self.eta * self.c * sum((cost - self.table.y) * self.table.x)
            costs[i] = self.c * sum((self.table.y * cost) + (1 - self.table.y) * cost) + (1 / 2) * sum(w.T**2)

        self.w = w
        self.b = b
        self.costs = costs
        self.graphics.append(Points(pts=[], color=self.color, isConnected=True))

    def predict(self, x_test):
        pred_y = []
        svm = x_test @ self.w.T + self.b
        for i in svm:
            if i >= 0:
                pred_y.append(1)
            else:
                pred_y.append(0)

        return pred_y

    def getY(self, _x, line=0):
        y = (line - self.b - _x * self.w.T[0]) / self.w.T[1]
        return y[0]

    def getX(self, _x, line=0):
        return (line - self.b - _x * self.w.T[1]) / self.w.T[0]


if __name__ == '__main__':
    hp.clear()
    print("Running Models MAIN")
    table = Table(filePath="examples/decisionTree/small").createXXYTable()
    train, test = table.partition()
    knn = KNN(table=train, partition=0.8, k=3)
    print("Accuracy:", 100 * knn.accuracy(testTable=test))
    for _x in knn.x:
        print(_x, end=" Closest: ")
        for j in knn.getNeighbor(_x):
            print(knn.x[j], knn.y[j], end=" | ")
        print()
