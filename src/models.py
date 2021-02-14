# from sklearn import linear_model
from graphics import Color
from sklearn import metrics
from scipy import spatial
import numpy as np
import helper as hp
from graphics import Points, HStack
from math import inf, sqrt, log, pi, sin, cos, e
# from random import uniform


class Model:
    def __init__(self, table, testingTable=None, name="", color=Color.red, isUserSet=False, isClassification=False, isRegression=False, **kwargs):
        self.name = name
        self.color = color
        # isLinear=False, isCategorical=False, isConnected=False, displayRaw=False
        # self.isLinear = isLinear
        # self.isCategorical = isCategorical
        # self.isConnected = isConnected
        # self.displayRaw = displayRaw
        self.graphics = []  # stores Point objects
        self.graphicsDict = {}
        self.isUserSet = isUserSet
        self.isRunning = False
        self.isClassification = isClassification
        self.isRegression = isRegression
        self.setTable(table=table, testingTable=testingTable, isReset=False)

    def setTable(self, table, testingTable=None, isReset=True):
        self.table = table
        self.testingTable = testingTable
        self.minX1, self.maxX1 = self.table.minX1, self.table.maxX1
        self.minX2, self.maxX2 = self.table.minX2, self.table.maxX2
        if isReset:
            self.reset()

    def getLinearPts(self, isLinear=True, stripeCount=False, m=None, b=None, **kwargs):
        edgePts = [
            (self.minX1, self.getY(self.minX1, **kwargs)),
            (self.maxX1, self.getY(self.maxX1, **kwargs)),
            (self.getX(self.minX2, **kwargs), self.minX2),
            (self.getX(self.maxX2, **kwargs), self.maxX2),
        ]
        if isLinear:
            if not stripeCount:
                return [self.table.getPt(x, y, self.color) for x, y in edgePts if(x >= self.minX1 and x <= self.maxX1 and y >= self.minX2 and y <= self.maxX2)]
            edgePts = sorted([(x, y) for x, y in edgePts if(x >= self.minX1 and x <= self.maxX1 and y >= self.minX2 and y <= self.maxX2)])

            x, y = edgePts[0]
            endX, endY = edgePts[1]

            stripeLength = (endX - x) / (2 * stripeCount)
            dx = max(hp.quad(
                1 + m * m,
                -2 * (x + m * (y - b)),
                x * x + (y - b) * (y - b) - stripeLength * stripeLength
            )) - x

            alt = 1
            pts = [self.table.getPt(x, y, self.color)]

            x += dx
            y = self.getY(x, **kwargs)
            while x < endX:
                pts.append(self.table.getPt(x, y, self.color) if alt < 3 else None)
                x += dx
                y = self.getY(x, **kwargs)
                alt = (alt + 1) % 4
            if alt < 3:
                pts.append(self.table.getPt(endX, endY, self.color) if alt else None)
            return pts

        edges = []
        for x, y in edgePts:
            if type(x) != tuple:
                x = [x]
            if type(y) != tuple:
                y = [y]
            edges += [(x_, y_) for x_ in x for y_ in y if x_ != None and y_ != None and x_ >= self.minX1 and x_ <= self.maxX1 and y_ >= self.minX2 and y_ <= self.maxX2]
        edgePts = edges
        edgePts.sort()

        out = []
        for i in range(1, len(edgePts), 2):
            # sweep
            out += self.getSweepingPts(start=edgePts[i - 1][0], end=edgePts[i][0], count=20)
        return out

    def getSweepingPts(self, start=None, end=None, count=40):
        if start == None:
            start = self.minX1
        if end == None:
            end = self.maxX1
        return [self.table.getPt(num, self.getY(num), self.color) for num in hp.rangx(start, end, (end - start) / count, outputEnd=True)]

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
        pass
        # raise NotImplementedError("Please Implement reset")

    def getY(self, x):
        raise NotImplementedError("Please Implement getY")

    def getX(self, y):
        raise NotImplementedError("Please Implement getX")


class Classifier(Model):
    # takes multiple features and outputs a categorical data
    def __init__(self, **kwargs):
        super().__init__(isLinear=False, isConnected=False, isClassification=True, **kwargs)
        self.colNameA, self.colNameB = self.table.xNames[0], self.table.xNames[1]

    def accuracy(self, testTable):
        if testTable == None:
            raise Exception("Model cannot find accuracy if testTable is None")
        return self.run_accuracy(testTable.x, testTable.y)

    def run_accuracy(self, x, y):
        correct = 0
        count = len(x)
        for i in range(count):
            correct += 1 if self.predict(x[i]) == y[i] else 0
        return correct / count

    def predict(self, _x):
        pass

    def getScoreString(self):
        return self.name + " Acc: " + str(round(100 * self.accuracy(testTable=self.testingTable), 2)) + "%"

    def defaultScoreString(self):
        return self.name + " Acc: --"

    def getCircleLabelPts(self, table=None):  # circle
        if table == None:
            table = self.getTable()

        rowCount = table.rowCount
        if rowCount > 0:
            trig = 2.0 * pi / rowCount

        pts = []
        i = 0
        for label in table.y:
            pts.append((0.5 * cos(trig * i) if rowCount > 1 else 0.0,
                        0.5 * sin(trig * i) if rowCount > 1 else 0.0,
                        table.classColors[str(label[0])]))
            i += 1

        # if treeNode.parent != None:
        #     items.append(Label(text="{}:{}".format(treeNode.parent.column, treeNode.value), fontSize=20, color=Color.white, dx=-0.95, dy=-1))
        # return ZStack(items=items, keywords="dotStack", limit=150)
        return pts


class Regression(Model):
    # takes multiple features and outputs real data

    def __init__(self, length, **kwargs):
        super().__init__(isConnected=True, isRegression=True, **kwargs)
        self.length = length
        self.colNameA, self.colNameB = self.table.xNames[0], self.table.yName
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

    def getEqString(self):
        raise NotImplementedError("Please Implement getEqString")

    def getEq(self):
        raise NotImplementedError("Please Implement getEq")

    def getScoreString(self):
        return self.name + " Error: " + str(round(self.error(testTable=self.testingTable), 4))

    def defaultScoreString(self):
        return self.name + " Error: --"

    def cefString(self, constant, power, showPlus=True, roundValue=2):
        while type(constant) == np.ndarray:
            constant = constant[0]
        constant = round(constant, roundValue)
        if constant == 0:
            return "0"
        return ("+" if constant > 0 and showPlus else "") + str(constant) + ("" if power <= 0 else ("x" + ("" if power == 1 else hp.superscript(power))))

    def getPts(self, start=None, end=None, count=40):  # get many points
        return self.getSweepingPts(start=start, end=end, count=count)

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
        self.graphics.append(Points(pts=self.getCircleLabelPts(), color=self.color, isConnected=False))

    def setTable(self, **kwargs):
        super().setTable(**kwargs)
        self.curr = DTNode(table=self.table)

    def getTable(self):
        return self.curr.table

    def getChildren(self):
        return self.curr.children

    def getParent(self):
        return self.curr.parent

    def getParentColumn(self):
        return self.curr.parent.colIndex

    def getParentColName(self):
        return self.curr.parent.getColName()

    def getColName(self):
        return self.curr.getColName()

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
            newTable = self.table.matchValue(colIndex=self.colIndex, value=item)
            self.children.append(DTNode(
                table=newTable,
                parent=self, value=item
            ))

    def reset(self):
        self.colIndex = None
        self.children = []

    def predict(self, x):
        # print("Row:", x)
        if len(self.children) > 0:
            for child in self.children:
                if child.value == x[self.colIndex]:
                    return child.predict(x)
            # return None
        # print("Majority:", self.table.majorityInTargetColumn())
        return self.table.majorityInTargetColumn()

    def getColName(self):
        return self.table.xNames[self.colIndex] if self.colIndex != None else None


class RandomForest(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setTable(self, **kwargs):
        super().setTable(**kwargs)
        self.curr = self.newTree()
        self.trees = []

    def newTree(self):
        return DecisionTree(table=self.table, testingTable=self.testingTable)

    def predict(self, _x):
        mostFreq = None
        predictions = {}
        predictions[mostFreq] = 0

        for tree in self.trees:
            p = tree.predict(_x)
            if(p in predictions):
                predictions[p] += 1
            else:
                predictions[p] = 1

            if predictions[p] > predictions[mostFreq]:
                mostFreq = p
        return mostFreq

    def saveTree(self):
        self.trees.append(self.curr)
        self.curr = self.newTree()

    def add(self, *args, **kwargs):
        return self.curr.add(*args, **kwargs)

    def getParent(self, *args, **kwargs):
        return self.curr.getParent(*args, **kwargs)

    def getChildren(self, *args, **kwargs):
        return self.curr.getChildren(*args, **kwargs)

    def getColName(self, *args, **kwargs):
        return self.curr.getColName(*args, **kwargs)

    def getChild(self, *args, **kwargs):
        return self.curr.getChild(*args, **kwargs)

    def remove(self, *args, **kwargs):
        return self.curr.remove(*args, **kwargs)

    def hasChildren(self, *args, **kwargs):
        return self.curr.hasChildren(*args, **kwargs)

    def go(self, *args, **kwargs):
        return self.curr.go(*args, **kwargs)

    def getParentColName(self, *args, **kwargs):
        return self.curr.getParentColName(*args, **kwargs)

    def getValue(self, *args, **kwargs):
        return self.curr.getValue(*args, **kwargs)

    def isRoot(self, *args, **kwargs):
        return self.curr.isRoot(*args, **kwargs)

    def goBack(self, *args, **kwargs):
        return self.curr.goBack(*args, **kwargs)

    def getTable(self, *args, **kwargs):
        return self.curr.getTable(*args, **kwargs)

    def __len__(self):
        return len(self.trees)


class KNN(Classifier):

    def __init__(self, k=3, bestK=False, **kwargs):
        self.k = k
        self.bestK = bestK
        super().__init__(displayRaw=True, **kwargs)

    def setTable(self, **kwargs):
        super().setTable(**kwargs)
        self.kdTree = spatial.KDTree(np.array(self.table.x))
        if self.bestK:
            self.findBestK(testTable=self.testingTable)

    def getNeighbor(self, _x):
        return np.reshape(self.kdTree.query(_x, k=self.k)[1], self.k)  # if type(_x) == np.ndarray else np.array(_x)

    def predict(self, _x):
        ys = np.array([self.table.y[i][0] for i in self.getNeighbor(_x)])
        # return np.argmax(np.bincount(ys))
        u, indices = np.unique(ys, return_inverse=True)
        return u[np.argmax(np.bincount(indices))]

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

    def reset(self):
        if "n" in self.__dict__:
            self.length = self.n + 1
        super().reset()

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
            # print("DJ:", self.dJ, "CEF:", self.cef)
            ptsGraphics = self.getGraphic("pts")
            if ptsGraphics != None:
                ptsGraphics.setPts(self.getPts())

            eqGraphics = self.getGraphic("eq")
            if eqGraphics != None:
                eqGraphics.setFont(text=self.getEqString())

            errGraphics = self.getGraphic("err")
            if errGraphics != None:
                errGraphics.setFont(text=self.getScoreString())
            return
        print("FIT DONE")
        self.isRunning = False

    def getJ(self, cef):
        total = 0
        # testTotal = 0
        for i in range(self.table.rowCount):
            localTotal = self.table.y[i][0]
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
            localTotal = -self.table.y[i][0]
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
            out += self.cefString(constant=self.cef[i], power=n, showPlus=i > 0, roundValue=4)
            n -= 1
        return out

    def getPts(self):
        return self.getLinearPts(isLinear=self.n == 1)


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
            exp = e**(x * self.cef[0] + self.cef[1])
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


class SVM(Classifier):
    def __init__(self, C=0.005, n_iters=10000, learning_rate=0.000001, **kwargs):
        super().__init__(length=0, **kwargs)
        self.c = C
        self.iter = n_iters
        self.eta = learning_rate
        self.reset()  # no reset in classifer init90
        # print("X Shape:", self.table.x.shape)
        # print("W Shape:", self.w.shape)

    def setTable(self, **kwargs):
        super().setTable(**kwargs)
        self.data = {-1: [], 1: []}
        for i in range(self.table.rowCount):
            self.data[self.table.y[i][0]].append(self.table.x[i])
        self.opt_dict = {}
        self.transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        self.all_data = np.array([])
        for yi in self.data:
            self.all_data = np.append(self.all_data, self.data[yi])
        self.max_feature_value = max(self.all_data)
        self.min_feature_value = min(self.all_data)
        self.all_data = None

        # with smaller steps our margins and db will be more precise
        self.step_sizes = [self.max_feature_value * 0.1,
                           self.max_feature_value * 0.01,
                           # point of expense
                           self.max_feature_value * 0.001, ]

        # extremly expensise
        self.b_range_multiple = 5
        # we dont need to take as small step as w
        self.b_multiple = 5

        self.latest_optimum = self.max_feature_value * 10
        self.stepIndex = 0

    def reset(self):
        self.w = np.zeros([1, self.table.x.shape[1]])
        self.b = 0
        self.costs = np.zeros(self.iter)
        self.counter = 0

        self.reg_strength = 10000

    def fit(self):
        if self.stepIndex >= len(self.step_sizes):
            self.isRunning = False
            print("Training Done")
            return
        step = self.step_sizes[self.stepIndex]
        self.stepIndex += 1

        w = np.array([self.latest_optimum, self.latest_optimum])

        # we can do this because convex
        optimized = False
        while not optimized:
            for b in np.arange(-1 * self.max_feature_value * self.b_range_multiple,
                               self.max_feature_value * self.b_range_multiple,
                               step * self.b_multiple):
                for transformation in self.transforms:
                    w_t = w * transformation
                    found_option = True

                    # weakest link in SVM fundamentally
                    # SMO attempts to fix this a bit
                    # ti(xi.w+b) >=1
                    for i in self.data:
                        for xi in self.data[i]:
                            yi = i
                            if not yi * (np.dot(w_t, xi) + b) >= 1:
                                found_option = False
                                break
                    if found_option:
                        """
                        all points in dataset satisfy y(w.x)+b>=1 for this cuurent w_t, b
                        then put w,b in dict with ||w|| as key
                        """
                        self.opt_dict[np.linalg.norm(w_t)] = [w_t, b]

            # after w[0] or w[1]<0 then values of w starts repeating itself because of transformation
            # Think about it, it is easy
            # print(w,len(self.opt_dict)) Try printing to understand
            if w[0] < 0:
                optimized = True
                # print("optimized a step")
            else:
                w = w - step

        # sorting ||w|| to put the smallest ||w|| at poition 0
        norms = sorted([n for n in self.opt_dict])
        # optimal values of w,b
        opt_choice = self.opt_dict[norms[0]]

        self.w = opt_choice[0]
        self.b = opt_choice[1]

        # start with new self.latest_optimum (initial values for w)
        self.latest_optimum = opt_choice[0][0] + step * 2

        self.getGraphic("acc").setFont(text=self.getScoreString())
        for i, pts in enumerate(self.getPts()):
            self.getGraphic("pts" + ("" if i == 0 else str(i + 1))).setPts(pts)

    def getY(self, x, v):
            # returns a x2 value on line when given x1
        return (-self.w[0] * x - self.b + v) / self.w[1]

    def getX(self, y, v):
        # returns a x1 value on line when given x2
        return (-self.w[1] * y - self.b + v) / self.w[0]

    def getPts(self, start=None, end=None, count=40):  # get many points
        return [self.getLinearPts(isLinear=True, v=v) for v in [0, -1, 1]]

    def predict(self, x):
        return 1 if (x @ self.w.T + self.b) >= 0 else -1
        # return [1 if i >= 0 else 0 for i in (x @ self.w.T + self.b)]


if __name__ == '__main__':
    hp.clear()
    print("Running Models MAIN")
    # from table import Table
    # table = Table(filePath="examples/decisionTree/small").createXXYTable()
    # train, test = table.partition()
    # knn = KNN(table=train, partition=0.8, k=3)
    # print("Accuracy:", 100 * knn.accuracy(testTable=test))
    # for _x in knn.x:
    #     print(_x, end=" Closest: ")
    #     for j in knn.getNeighbor(_x):
    #         print(knn.x[j], knn.y[j], end=" | ")
    #     print()
